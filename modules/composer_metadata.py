"""Text/resource metadata carried by cached Krea2 Composer conditioning."""

COMPOSER_CLASS = "Krea2TwoCharacterComposer"
METADATA_KEY = "cyberdelia_composer_metadata"


def cached_payloads(outputs):
    """Read output 1 only; output 5 is a human-readable diagnostic report."""
    if not isinstance(outputs, (list, tuple)) or len(outputs) < 2:
        return []
    found = []
    def visit(value, depth=0):
        if depth > 8:
            return
        if isinstance(value, dict):
            payload = value.get(METADATA_KEY)
            if (isinstance(payload, dict) and payload.get("version") == 1
                    and isinstance(payload.get("scene_prompt"), str)
                    and isinstance(payload.get("negative_prompt"), str)
                    and isinstance(payload.get("character_prompts"), list)
                    and len(payload["character_prompts"]) == 2
                    and all(isinstance(text, str) for text in payload["character_prompts"])):
                found.append(payload)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item, depth + 1)
    visit(outputs[1])
    return found


def positive_text(metadata):
    return "\n".join([f"Scene: {metadata.get('scene_prompt', '')}"] + [
        f"Character {n}: {text}" for n, text in enumerate(metadata.get("character_prompts", []), 1)
    ]).strip()


def loras(metadata):
    return [row for group in metadata.get("loras", []) if isinstance(group, list)
            for row in group if isinstance(row, dict) and isinstance(row.get("name"), str)
            and row["name"] and row["name"] != "None" and isinstance(row.get("strength"), (int, float))
            and row["strength"] != 0]


def details(metadata):
    result = {
        "Krea scene prompt": metadata.get("scene_prompt"),
        "Krea region layout": metadata.get("layout"),
        "Krea separation": metadata.get("separation"),
        "Krea LoRA edge softness": metadata.get("lora_edge_softness"),
        "Krea base strength": metadata.get("base_strength"),
        "Krea NegPiP": metadata.get("negpip"),
    }
    encoded_scene = metadata.get("encoded_scene_prompt")
    if isinstance(encoded_scene, str) and encoded_scene != metadata.get("scene_prompt"):
        result["Krea encoded scene prompt"] = encoded_scene
    for n, text in enumerate(metadata.get("character_prompts", []), 1):
        result[f"Krea character {n} prompt"] = text
    for n, rows in enumerate(metadata.get("loras", []), 1):
        if isinstance(rows, list):
            result[f"Krea character {n} LoRAs"] = "; ".join(
                f"{row['name']} @ {row['strength']:g}" for row in rows
                if isinstance(row, dict) and isinstance(row.get("name"), str)
                and isinstance(row.get("strength"), (int, float)))
    return result
