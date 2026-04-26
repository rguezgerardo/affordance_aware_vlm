import os
import json
import re
from typing import Dict, List, Any

import torch
from neo4j import GraphDatabase
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from test_neo4j import get_graph_context
from test_neo4j import get_mock_graph_context


MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


# load the qwen model
def load_qwen():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def qwen_infer(model, processor, image_path: str, prompt: str, max_new_tokens: int = 512) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0]


# extract json from qwen
def extract_json_from_text(text: str) -> Dict[str, Any]:
    # possible that Qwen may wrap JSON in markdown. This tries to extract the JSON object.
    text = text.strip()

    # remove markdown fences if present
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)

    # extract first JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in Qwen output:\n{text}")

    return json.loads(match.group(0))


def get_scene_facts(model, processor, image_path: str) -> Dict[str, Any]:
    prompt = """
Question: Is this safe?
Analyze the image and identify the main object in focus based on the prior question/prompt, its likely material, and any visible safety-relevant objects or conditions.

Use only these material labels:
["Plastic", "Metal", "Wood", "Brick", "Glass", "Cardboard", "Paper", "Ceramic", "Rubber", "Foam", "Flesh", "Unknown"]

Use only these condition labels:
["Extreme_Heat", "Fire", "Freezing_Cold", "Liquid_Water", "Electricity", "Heavy_Weight", "Sharp_Impact", "Microwave_Radiation", "Moderate_Heat", "Boiling_Water", "Steam", "Hot_Oil", "Prolonged_Sunlight", "Live_Electricity", "Sharp_Edge", "Puncture_Force", "Falling_Height", "Slippery_Surface", "Corrosive_Chemical", "Toxic_Chemical", "Food_Contact", "Child_Accessible", "Pet_Accessible", "Confined_Space", "Ventilation_Limited", "Moving_Machinery", "Trip_Hazard", "Choking_Size", "Unknown"]

Use only these object labels when applicable:
["Stove", "Oven", "Hot_Pan", "Kettle", "Pot", "Candle", "Match", "Lighter", "Knife", "Scissors", "Broken_Glass", "Hammer", "Nail", "Needle", "Outlet", "Live_Wire", "Microwave", "Bathtub", "Sink", "Rain", "Ice", "Snow", "Wet_Floor", "Oil_Spill", "Ladder", "Table_Edge", "Fan", "Loose_Cable", "Plastic_Bag", "Button_Battery", "Cleaning_Bottle", "Unknown"]

Return only a JSON object with this format:
{
  "main_object": "string",
  "material": "string",
  "nearby_objects": ["string"],
  "visible_conditions": ["string"],
  "spatial_relationships": [
    {
      "subject": "string",
      "relation": "on|near|inside|touching|under|above|unknown",
      "object": "string"
    }
  ],
  "confidence": {
    "material": "low|medium|high",
    "conditions": "low|medium|high",
    "spatial_relationships": "low|medium|high"
  }
}

Do not include explanations outside the JSON.
"""

    raw_output = qwen_infer(model, processor, image_path, prompt)
    return extract_json_from_text(raw_output)


# normalize qwen terms
MATERIAL_MAP = {
    "plastic": "plastic",
    "metal": "metal",
    "glass": "glass",
    "paper": "paper",
    "wood": "wood",
    "fabric": "fabric",
    "electronic": "electronics",
    "electronics": "electronics",
    "unknown": "unknown",
}

HAZARD_MAP = {
    "high heat": "high heat",
    "hot": "high heat",
    "hot pan": "high heat",
    "hot stove": "high heat",
    "stove": "high heat",
    "burner": "high heat",
    "flame": "flame",
    "fire": "flame",
    "water": "water",
    "wet": "water",
    "spill": "water",
    "direct sunlight": "direct sunlight",
    "sunlight": "direct sunlight",
    "impact": "impact",
    "fall": "impact",
    "instability": "instability",
    "unstable": "instability",
    "sharp object": "sharp object",
    "sharp": "sharp object",
}


def normalize_value(value: str, mapping: Dict[str, str]) -> str:
    if not value:
        return "unknown"

    value = value.lower().strip()

    for key, normalized in mapping.items():
        if key in value:
            return normalized

    return value


def normalize_scene(scene: Dict[str, Any]) -> Dict[str, Any]:
    materials = set()
    hazards = set()

    for obj in scene.get("objects", []):
        material = normalize_value(obj.get("material", ""), MATERIAL_MAP)
        if material != "unknown":
            materials.add(material)

        state = normalize_value(obj.get("state", ""), HAZARD_MAP)
        if state != "unknown":
            hazards.add(state)

    for hazard in scene.get("hazards", []):
        normalized_hazard = normalize_value(hazard, HAZARD_MAP)
        if normalized_hazard != "unknown":
            hazards.add(normalized_hazard)

    return {
        "materials": sorted(materials),
        "hazards": sorted(hazards),
        "scene_summary": scene.get("scene_summary", ""),
        "raw_scene": scene,
    }


# kg connection
class AffordanceKG:
    def __init__(self):
        # change this with your credentials
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD", "password")

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()

    def close(self):
        self.driver.close()

    def query_affordances(self, materials: List[str], hazards: List[str]) -> List[Dict[str, str]]:
        cypher = """
        MATCH (m:Material)-[:UNSAFE_WITH]->(h:Hazard)-[:CAUSES]->(o:Outcome)
        WHERE m.name IN $materials AND h.name IN $hazards
        RETURN m.name AS material, h.name AS hazard, o.name AS outcome
        """

        records, _, _ = self.driver.execute_query(
            cypher,
            materials=materials,
            hazards=hazards,
        )

        return [record.data() for record in records]


# final model answer with graph
def get_final_answer(
    model,
    processor,
    image_path: str,
    scene: Dict[str, Any],
    graph_context: Dict[str, Any],
) -> str:
    prompt = f"""
You are a physical safety reasoning assistant.

You are given:
1. The original image.
2. Scene information extracted from the image.
3. Structured knowledge graph facts about material hazards, inferred conditions, risks, and recommended actions.

Use the knowledge graph facts as the main source of physical reasoning.

Scene information:
{json.dumps(scene, indent=2)}

Knowledge graph facts:
{json.dumps(graph_context, indent=2)}

Return the final answer in this exact format:

Safety Label: safe / caution / unsafe / unknown

Main Reason:
One or two sentences explaining the most important physical risk.

Knowledge Graph Evidence:
Briefly mention the material, condition, relationship, risk, and action used.

Recommended Action:
One sentence.

Do not invent risks that are not supported by either the image or the knowledge graph.
"""

    return qwen_infer(model, processor, image_path, prompt, max_new_tokens=256)

# full pipeline (so far, 1 and 2 work on my local machine)
def run_pipeline(image_path: str):
    print("[1] Loading Qwen...")
    model, processor = load_qwen()

    print("[2] Extracting scene facts from image...")
    scene = get_scene_facts(model, processor, image_path)
    print("\nScene extracted by Qwen:")
    print(json.dumps(scene, indent=2))

    print("[3] Querying Neo4j knowledge graph...")
    graph_context = get_graph_context(scene) # should work now
    # try:
    #     graph_context = get_graph_context(scene)
    # except Exception as e:
    #     print(f"Neo4j failed, using mock KG context: {e}")
    #graph_context = get_mock_graph_context(scene)
    print("\nGraph context returned by KG:")
    print(json.dumps(graph_context, indent=2))

    print("[4] Asking Qwen for final safety judgment using KG context...")
    final_answer = get_final_answer(
        model=model,
        processor=processor,
        image_path=image_path,
        scene=scene,
        graph_context=graph_context,
    )

    print("\nFinal Answer:")
    print(final_answer)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()

    run_pipeline(args.image)