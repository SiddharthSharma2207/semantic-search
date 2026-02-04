import json
import os
from openai import OpenAI
from config import Settings
import json

settings = Settings()

client = OpenAI(
    api_key=settings.api_key,
    base_url=settings.base_url
)

NUM_VARIATIONS = 5


def generate_variations_llm(question: str, topic: str, n: int = NUM_VARIATIONS) -> list[str]:
    """
    Generates variations of a question using an LLM.
    """
    prompt = (
        f"Generate {n} diverse phrasing variations for the following question about '{topic}'. "
        f"The variations should have the same semantic meaning but differ in wording, tone, or structure. "
        f"Do not include the answer. Return only the questions, one per line.\n\n"
        f"Original Question: {question}\n"
    )

    system_prompt = """<role>
    You are a helpful assistant that generates paraphrased questions.
    </role>
    <instructions>
    - You will be given a question, topic and number of variations.
    - Generate diverse phrasing variations for the following question about topic.
    - The variations should have the same semantic meaning but differ in wording, tone, or structure.
    </instructions>
    <output_format>
    ```json
    {
    "variations": [List of questions],
    }
    ```
    </output_format>

    """

    try:
        response = client.chat.completions.create(
            model=settings.llm_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""<question>{question}</question>
                <topic>{topic}</topic>
                <num_variations>{n}</num_variations>"""}
            ],
            temperature=0.7,
        )

        output = response.choices[0].message.content.strip()
        result = output.split("```json")[-1].split("```")[0]
        json_result = json.loads(result)

        return json_result["variations"]

    except Exception as e:
        print(f"Error generating variations for '{question}': {e}")
        return [question]  # Fallback to original


def main():
    input_path = "data/faq.json"
    output_path = "data/generated_faq_llm.json"

    print(f"Reading from {input_path}...")
    try:
        with open(input_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {input_path}")
        return

    augmented_data = []

    for query_id,item in data.items():
        question = item["question"]
        answer = item["answer"]
        topic = item["topic"]


        print(f"Generating variations for: '{question}'...")
        variations = generate_variations_llm(question, topic, n=NUM_VARIATIONS)

        # Add original question too if not in variations (likely not, but good to ensure coverage)
        all_variations = list(set([question] + variations))

        print(f"  -> Generated {len(all_variations) - 1} new variations.")

        for var in all_variations:
            augmented_data.append({
                "question": var,
                "query_id": query_id,
                "topic": topic
            })

    print(f"Writing {len(augmented_data)} total items to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(augmented_data, f, indent=2)


if __name__ == "__main__":
    main()
