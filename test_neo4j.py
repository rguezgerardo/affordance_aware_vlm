import os
from neo4j import GraphDatabase


# ============================================================
# Neo4j configuration
# ============================================================


# change to the appropriate credentials
NEO4J_URI = "neo4j+s://b20ae93c.databases.neo4j.io"
NEO4J_AUTH = ("b20ae93c", "QTnkFOZZY3yOIarhieQ_quH4OP7HNrQeEj3KpgHTlcU")
NEO4J_DATABASE = "b20ae93c"

def get_neo4j_driver():
    return GraphDatabase.driver(
        NEO4J_URI,
        auth=NEO4J_AUTH,
    )


# ============================================================
# Query 1: Infer conditions from detected objects
# Example: Stove -> Extreme_Heat
# ============================================================

def infer_conditions_from_objects(driver, object_names):
    object_names = [
        obj for obj in object_names
        if obj and obj != "Unknown"
    ]

    if not object_names:
        return []

    query = """
    MATCH (o:Object)-[:IMPLIES]->(c:Condition)
    WHERE o.name IN $object_names
    RETURN DISTINCT
           o.name AS object,
           c.name AS condition
    ORDER BY object, condition
    """

    records, _, _ = driver.execute_query(
        query,
        {"object_names": object_names},
        database_=NEO4J_DATABASE,
    )

    return [record.data() for record in records]


# ============================================================
# Query 2: Find material-condition hazards
# Example: Plastic + Extreme_Heat -> MELTS_AT
# ============================================================

def query_material_condition_hazards(driver, material, conditions):
    if not material or material == "Unknown":
        return []

    conditions = [
        c for c in conditions
        if c and c != "Unknown"
    ]

    if not conditions:
        return []

    query = """
    MATCH (m:Material {name: $material})-[r]->(c:Condition)
    WHERE c.name IN $conditions
    RETURN m.name AS material,
           type(r) AS relationship,
           c.name AS condition,
           r.severity AS severity,
           r.safety_label AS safety_label,
           r.explanation AS explanation,
           r.recommended_action AS recommended_action
    ORDER BY condition, relationship
    """

    records, _, _ = driver.execute_query(
        query,
        {
            "material": material,
            "conditions": conditions,
        },
        database_=NEO4J_DATABASE,
    )

    return [record.data() for record in records]


# ============================================================
# Query 3: Find condition-condition hazards
# Example: Liquid_Water -> INCREASES_RISK_OF -> Electricity
# ============================================================

def query_condition_condition_hazards(driver, conditions):
    conditions = [
        c for c in conditions
        if c and c != "Unknown"
    ]

    if not conditions:
        return []

    query = """
    MATCH (c1:Condition)-[r]->(c2:Condition)
    WHERE c1.name IN $conditions
       OR c2.name IN $conditions
    RETURN c1.name AS source_condition,
           type(r) AS relationship,
           c2.name AS target_condition,
           r.severity AS severity,
           r.safety_label AS safety_label,
           r.explanation AS explanation,
           r.recommended_action AS recommended_action
    ORDER BY source_condition, relationship, target_condition
    """

    records, _, _ = driver.execute_query(
        query,
        {"conditions": conditions},
        database_=NEO4J_DATABASE,
    )

    return [record.data() for record in records]


# ============================================================
# Query 4: Find risks and recommended actions
# Example: Plastic + Extreme_Heat -> Melting_Risk -> Move_Away
# ============================================================

def query_risks_and_actions(driver, material, conditions):
    conditions = [
        c for c in conditions
        if c and c != "Unknown"
    ]

    if not conditions:
        return []

    query = """
    OPTIONAL MATCH (m:Material {name: $material})-[mr:HAS_RISK]->(risk1:Risk)
    WHERE mr.condition IN $conditions
    OPTIONAL MATCH (risk1)-[:RECOMMENDS]->(action1:Action)

    WITH collect(DISTINCT {
        source: $material,
        source_type: "Material",
        condition: mr.condition,
        risk: risk1.name,
        action: action1.name
    }) AS material_risk_rows

    OPTIONAL MATCH (c:Condition)-[:HAS_RISK]->(risk2:Risk)
    WHERE c.name IN $conditions
    OPTIONAL MATCH (risk2)-[:RECOMMENDS]->(action2:Action)

    WITH material_risk_rows,
         collect(DISTINCT {
            source: c.name,
            source_type: "Condition",
            condition: c.name,
            risk: risk2.name,
            action: action2.name
         }) AS condition_risk_rows

    WITH material_risk_rows + condition_risk_rows AS rows

    UNWIND rows AS row
    WITH row
    WHERE row.risk IS NOT NULL

    RETURN DISTINCT
           row.source AS source,
           row.source_type AS source_type,
           row.condition AS condition,
           row.risk AS risk,
           row.action AS recommended_action
    ORDER BY risk, recommended_action
    """

    records, _, _ = driver.execute_query(
        query,
        {
            "material": material,
            "conditions": conditions,
        },
        database_=NEO4J_DATABASE,
    )

    return [record.data() for record in records]
# ============================================================
# Combined graph context function
# ============================================================

def get_graph_context(scene):
    """
    Expected scene format:

    {
      "main_object": "plastic container",
      "material": "Plastic",
      "nearby_objects": ["Stove"],
      "visible_conditions": ["Extreme_Heat"],
      "spatial_relationships": [
        {
          "subject": "plastic container",
          "relation": "on",
          "object": "Stove"
        }
      ],
      "confidence": {
        "material": "high",
        "conditions": "medium",
        "spatial_relationships": "high"
      }
    }
    """

    material = scene.get("material", "Unknown")
    nearby_objects = scene.get("nearby_objects", [])
    visible_conditions = scene.get("visible_conditions", [])

    with get_neo4j_driver() as driver:
        object_condition_facts = infer_conditions_from_objects(
            driver,
            nearby_objects,
        )

        inferred_conditions = [
            fact["condition"]
            for fact in object_condition_facts
            if fact.get("condition")
        ]

        all_conditions = sorted(
            set(visible_conditions + inferred_conditions)
        )

        material_hazards = query_material_condition_hazards(
            driver,
            material,
            all_conditions,
        )

        condition_hazards = query_condition_condition_hazards(
            driver,
            all_conditions,
        )

        risks_and_actions = query_risks_and_actions(
            driver,
            material,
            all_conditions,
        )

    return {
        "material": material,
        "visible_conditions": visible_conditions,
        "inferred_conditions_from_objects": object_condition_facts,
        "all_conditions_used_for_lookup": all_conditions,
        "material_condition_hazards": material_hazards,
        "condition_condition_hazards": condition_hazards,
        "risks_and_actions": risks_and_actions,
    }

def get_mock_graph_context(scene):
    return {
        "material": scene.get("material", "Unknown"),
        "visible_conditions": scene.get("visible_conditions", []),
        "inferred_conditions_from_objects": [
            {"object": "Stove", "condition": "Extreme_Heat"}
        ],
        "all_conditions_used_for_lookup": ["Extreme_Heat"],
        "material_condition_hazards": [
            {
                "material": "Plastic",
                "relationship": "MELTS_AT",
                "condition": "Extreme_Heat",
                "severity": "high",
                "safety_label": "unsafe",
                "explanation": "Plastic can melt when exposed to extreme heat.",
                "recommended_action": "Move the plastic away from the heat."
            }
        ],
        "condition_condition_hazards": [],
        "risks_and_actions": [
            {
                "source": "Plastic",
                "source_type": "Material",
                "condition": "Extreme_Heat",
                "risk": "Melting_Risk",
                "recommended_action": "Move_Away"
            }
        ]
    }


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    scene = {
        "main_object": "plastic container",
        "material": "Plastic",
        "nearby_objects": ["Stove"],
        "visible_conditions": [],
        "spatial_relationships": [
            {
                "subject": "plastic container",
                "relation": "on",
                "object": "Stove",
            }
        ],
        "confidence": {
            "material": "high",
            "conditions": "medium",
            "spatial_relationships": "high",
        },
    }

    graph_context = get_graph_context(scene)

    print(graph_context)