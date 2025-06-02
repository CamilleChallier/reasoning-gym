#!/usr/bin/env python
"""
Configuration generator for reasoning-gym evaluation.

This script generates a YAML configuration file with all registered datasets
from reasoning_gym, organized by category.

Usage:
    python generate_config.py [options]

Options:
    --output OUTPUT       Output YAML file path (default: all_datasets.yaml)
    --model MODEL         Model name (default: openai/gpt-4)
    --provider PROVIDER   Provider name (default: None)
    --size SIZE           Default dataset size (default: 100)
    --seed SEED           Default dataset seed (default: 42)
    --include-params      Include all configuration parameters (default: False)
    --category CATEGORY   Only include datasets from this category (default: None)
"""

import argparse
import inspect
from collections import defaultdict
from dataclasses import fields

import yaml

from reasoning_gym.factory import DATASETS, CURRICULA


def extract_category(module_name):
    """Extract category from module name."""
    parts = module_name.split(".")
    if len(parts) >= 3:
        return parts[1]  # reasoning_gym.{category}.dataset_name
    return "other"


def generate_config(model, provider, size, seed, include_params, category=None):
    """Generate configuration with all registered datasets.

    Args:
        model: Model name
        provider: Provider name
        size: Default dataset size
        seed: Default dataset seed
        include_params: Whether to include all configuration parameters
        category: If specified, only include datasets from this category
    """
    # Group datasets by category
    categories = defaultdict(list)

    for dataset_name, (dataset_cls, config_cls) in DATASETS.items():
        # Extract category from module name
        dataset_category = extract_category(dataset_cls.__module__)

        # Skip if a specific category was requested and this doesn't match
        if category and dataset_category != category:
            continue

        # Create dataset entry
        dataset_entry = {"dataset": dataset_name}

        # Optionally include all configuration parameters
        if include_params:
            params = {}
            # Get default values from config class fields
            for field in fields(config_cls):
                # Skip seed and size as they're handled separately
                if field.name not in ["seed", "size"]:
                    # Only include fields with default values
                    if field.default != inspect.Parameter.empty:
                        params[field.name] = field.default

            if params:
                dataset_entry["params"] = params

        # Add to appropriate category
        categories[dataset_category].append(dataset_entry)

    # Create configuration structure
    config = {
        "model": model,
        "provider": provider,
        "output_dir": "results",
        "max_concurrent": 10,
        "default_size": size,
        "default_seed": seed,
        "categories": [],
    }

    # Add categories
    for category_name, datasets in sorted(categories.items()):
        config["categories"].append({"category": category_name, "datasets": datasets})

    return config

def convert_to_yaml_safe(obj):
    """Recursively convert tuples to lists to avoid YAML unsafe tags."""
    if isinstance(obj, dict):
        return {k: convert_to_yaml_safe(v) for k, v in obj.items()}
    elif isinstance(obj, tuple):
        return list(convert_to_yaml_safe(v) for v in obj)
    elif isinstance(obj, list):
        return [convert_to_yaml_safe(v) for v in obj]
    else:
        return obj
    
def generate_level_config(model, provider, size, seed, category=None, level: int = 0):
    categories = defaultdict(list)

    for dataset_name, (dataset_cls, config_cls) in DATASETS.items():
        dataset_category = extract_category(dataset_cls.__module__)

        if category and dataset_category != category:
            continue

        if dataset_name not in CURRICULA:
            print(f"No curriculum registered for dataset '{dataset_name}'")
            continue

        curriculum_cls = CURRICULA[dataset_name]
        curriculum = curriculum_cls()
        curriculum.set_global_level(level)
        config_instance = curriculum.generate_configuration()

        dataset_entry = {"dataset": dataset_name}

        # Serialize config dataclass to dictionary, but only include fields controlled by curriculum
        from reasoning_gym.coaching import BaseCurriculum, ScalarAttributeDefinition, RangeAttributeDefinition
        params = {}
        curriculum_fields = set()
        for attr in curriculum.attributes.values():
            if isinstance(attr, ScalarAttributeDefinition):
                curriculum_fields.add(attr.field_name)
            elif isinstance(attr, RangeAttributeDefinition):
                curriculum_fields.add(attr.lower_field_name)
                curriculum_fields.add(attr.upper_field_name)


        for field in fields(config_cls):
            if field.name not in ["seed", "size"] and field.name in curriculum_fields:
                value = getattr(config_instance, field.name, None)
                if value is not None:
                    params[field.name] = convert_to_yaml_safe(value)

        if params:
            dataset_entry["params"] = params
        categories[dataset_category].append(dataset_entry)

    # Assemble the full config dictionary
    config = {
        "model": model,
        "provider": provider,
        "output_dir": "results",
        "max_concurrent": 10,
        "default_size": size,
        "default_seed": seed,
        "categories": [],
    }

    for cat_name, datasets in categories.items():
        config["categories"].append({
            "category": cat_name,
            "datasets": datasets
        })

    return config




def main():
    parser = argparse.ArgumentParser(description="Generate evaluation configuration with all datasets")
    parser.add_argument("--output", default="all_datasets.yaml", help="Output YAML file path")
    parser.add_argument("--model", default="openai/gpt-4", help="Model name")
    parser.add_argument("--provider", default=None, help="Provider name")
    parser.add_argument("--size", type=int, default=100, help="Default dataset size")
    parser.add_argument("--seed", type=int, default=42, help="Default dataset seed")
    parser.add_argument("--include-params", action="store_true", help="Include all configuration parameters")
    parser.add_argument("--category", help="Only include datasets from this category")
    parser.add_argument("--level", type=int, help="Curriculum level to use")

    args = parser.parse_args()

    # Generate configuration
    config = generate_level_config(
        model=args.model,
        provider=args.provider,
        size=args.size,
        seed=args.seed,
        # include_params=args.include_params,
        category=args.category,
        level=args.level,
    )

    # Write to file
    with open(args.output, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # print(
    #     f"Configuration with {sum(len(cat['datasets']) for cat in config['categories'])} datasets written to {args.output}"
    # )
    # print(f"Categories: {', '.join(cat['category'] for cat in config['categories'])}")


if __name__ == "__main__":
    main()
