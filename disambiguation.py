"""
Interpretation generation for CLUES framework.

Provides two modes of disambiguation:
1. generate_question_interpretations: Guided disambiguation with domain constraints.
2. generate_explore_meanings: Stage 1 diversity for uncertainty estimation.

Paper Reference:
    Section 4.1: Resolving Ambiguity in Epidemiological Questions.
"""

import logging
from typing import Optional

from clues.llm import GeminiClient
from clues.schemas import DisambiguationResponse, ExploreMeaningsResponse

logger = logging.getLogger(__name__)


async def generate_question_interpretations(
    question: str,
    num_interpretations: Optional[int] = None,
    client: Optional[GeminiClient] = None,
) -> DisambiguationResponse:
    """
    Generate multiple interpretations of an ambiguous question (guided disambiguation).

    Uses domain-specific guidelines for epidemiological research to produce
    clinically plausible interpretations.

    Args:
        question: Original user question.
        num_interpretations: If specified, generate exactly this many interpretations.
        client: GeminiClient instance. If None, creates one.

    Returns:
        DisambiguationResponse with interpretations and ambiguity score.
    """
    if client is None:
        client = GeminiClient()

    disambiguation_prompt = """
    # Disambiguation for Text-to-SQL Questions

    You are a disambiguation assistant for a text-to-SQL system. Analyze input text
    and identify potential ambiguities that could lead to different SQL interpretations.

    ## Your Task:
    1. Analyze the input text for ambiguities related to:
       - Patient counts (unique patients vs all records)
       - Temporal relationships (before/after, during, within timeframes)
       - Population definitions (inclusion/exclusion criteria)
       - Demographic specifications (age calculations, gender)
       - Measurement specifications (values, units, thresholds)
       - Event ordering (first vs. any occurrence)

    2. For each ambiguity, explain how it affects the query structure.

    3. Calculate an overall ambiguity score from 0 to 1:
       - 0: Completely unambiguous
       - 0.1-0.3: Slightly ambiguous
       - 0.4-0.6: Moderately ambiguous
       - 0.7-0.9: Highly ambiguous
       - 1.0: Completely ambiguous
    """

    base_instructions = """
       - Each interpretation must resolve every identified ambiguity
       - Each interpretation should be a complete, unambiguous text
       - Order from more plausible (first) to less plausible (last)
       - Make each interpretation specific enough to generate a precise SQL query
    """

    if num_interpretations is not None:
        interpretation_instructions = f"""
    4. Generate EXACTLY {num_interpretations} final interpretations that:
       - Cover the most meaningful combinations of identified ambiguities
       {base_instructions}
       - If unambiguous, return the same interpretation multiple times.
        """
    else:
        interpretation_instructions = f"""
    4. Generate a comprehensive set of final interpretations that:
       - Covers ALL meaningful combinations of identified ambiguities
       {base_instructions}
        """

    guidelines = """
    ## Guidelines:
    - When "patient counts" are mentioned, interpret as UNIQUE patients unless stated otherwise
    - Pay attention to temporal relationships and event ordering
    - If no timeframe is defined, include the whole period available
    - Do not include any concept_id or table names
    - Do not mention specific medical codes (ICD-10, SNOMED, etc.)
    """

    analysis_instruction = "# Now analyze the following text:\n\n"
    if num_interpretations is not None:
        analysis_instruction = f"# Now analyze the following text (generate EXACTLY {num_interpretations} interpretations):\n\n"

    combined_prompt = (
        f"{disambiguation_prompt}\n{interpretation_instructions}\n{guidelines}\n\n"
        f"{analysis_instruction}{question}"
    )

    response = await client.get_response(
        prompt=combined_prompt,
        response_schema=DisambiguationResponse,
    )

    return response


async def generate_explore_meanings(
    question: str,
    num_interpretations: int = 10,
    client: Optional[GeminiClient] = None,
) -> ExploreMeaningsResponse:
    """
    Stage 1: Generate diverse semantic interpretations for uncertainty estimation.

    Simple exploratory prompt designed to capture the full breadth of potential
    ambiguity WITHOUT domain constraints. Used for H(I) calculation.

    Args:
        question: Original user question.
        num_interpretations: Number of interpretations to generate (default 10).
        client: GeminiClient instance. If None, creates one.

    Returns:
        ExploreMeaningsResponse with diverse interpretations.
    """
    if client is None:
        client = GeminiClient()

    prompt = f"""You are a **clinical data analyst** whose job is to translate a user's question into specific, executable queries for a large clinical database.
    Your task is to generate {num_interpretations} plausible and specific interpretations of the following question.

**User Question:**
"{question}"

## Instructions:
1. Analyze the question for any potential ambiguities (in concepts, timeframes, cohorts, etc.).
2. Generate {num_interpretations} specific interpretations that could each correspond to a different database query.
3. If the question is ambiguous, explore different valid ways to resolve the ambiguities.
4. If the question is already clear, it is acceptable for interpretations to be paraphrases of the same meaning.
5. Never mention any ontology or medical codes (SNOMED, ICD10, LOINC).

## What NOT to Do:
- Do not invent new constraints if the original question is clear.
- Do not propose interpretations requiring external data sources.
- Do not specify the data source. Assume all queries run against the same database.
- Do not number or label your interpretations.

Generate {num_interpretations} plausible interpretations. Return ONLY the interpretation text."""

    response = await client.get_response(
        prompt=prompt,
        response_schema=ExploreMeaningsResponse,
    )

    # Ensure we have exactly num_interpretations
    if len(response.interpretations) < num_interpretations:
        logger.warning(
            f"Only got {len(response.interpretations)} interpretations, "
            f"expected {num_interpretations}. Padding with original."
        )
        while len(response.interpretations) < num_interpretations:
            response.interpretations.append(question)
    elif len(response.interpretations) > num_interpretations:
        response.interpretations = response.interpretations[:num_interpretations]

    return response
