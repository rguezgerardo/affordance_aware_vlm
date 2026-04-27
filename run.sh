#!/bin/bash

for img in images/*; do
    python3 vlm_kg_pipeline.py --image "$img"
done