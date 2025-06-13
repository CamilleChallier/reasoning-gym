from dataclasses import dataclass
from random import Random
from typing import Any, Optional

from ..coaching import BaseCurriculum, ScalarAttributeDefinition
from ..data.wordle_words import wordle_words
from ..factory import ProceduralDataset, register_dataset
from .contrib.bfit.Compiler import Compiler, Minify

import bfi

DATASET_NAME = "bf"


@dataclass
class BFConfig:
    """Configuration for BF task generation"""

    seed: Optional[int] = None
    size: int = 500
    difficulty: int = 0

    def validate(self) -> None:
        """Validate configuration parameters"""
        assert 0<= self.difficulty < 4, "difficulty must be less than 4"


class BFDataset(ProceduralDataset):
    """Generates BF tasks"""

    def __init__(self, config: BFConfig):
        self._prompt_templates = ( "This is a Brainf*ck (BF) program. BF is a minimalistic language with only 8 commands:\n\n"
                                    "> : Move the pointer one cell to the right\n"
                                    "< : Move the pointer one cell to the left\n"
                                    "+ : Increment the current cell by 1\n"
                                    "- : Decrement the current cell by 1\n"
                                    "[ : Jump forward to the matching ] if the current cell is 0\n"
                                    "] : Jump back to the matching [ if the current cell is not 0\n"
                                    ". : Output the ASCII character at the current cell\n"
                                    ", : Read a single character of input into the current cell\n\n"
                                    "Here is an example:\n"
                                    "BF code: `++++++++[>++++++++<-]>+.`\n"
                                    "This builds the number 65 (ASCII \"A\") and prints it.\n"
                                    "Output: `A`\n\n"
                                    "Now, what is the output of the following BF program?\n\n"
                                    "{bf_program}\n\n"
                                    "Respond only with the exact output of the program."
                                )

        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict:
        """Generate a single BF task

        Returns:
            dict with keys:
                - question: str, the task description with BF program
                - answer: str, the result of this BF program BFI execution
                - metadata: dict with generation parameters
        """
        rng = Random(self.seed + idx)

        bfit_code = self.generate_bfit_code(self.config.difficulty, rng)
        bf_program = self.compile_bfit_code_to_bf(bfit_code)
        result = bfi.interpret(bf_program, buffer_output=True)

        return {
            "question": self._prompt_templates.format(bf_program=bf_program),
            "answer": result,
            "metadata": {
                "source_dataset": DATASET_NAME,
                "source_index": idx,
                "bfit_code": bfit_code,
                "bf_program": bf_program,
                "difficulty": {"difficulty": self.config.difficulty},
            },
        }

    def generate_bfit_code(self, difficulty, rng: Random) -> str:
        
        if difficulty == 0:
            # Output a single letter
            letter = rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            bfit_template = f"""
    int main() {{
        print("{letter}");
    }}
    """

        elif difficulty == 1:
            word = rng.choice(wordle_words)
            bfit_template = f"""
int main() {{
    print("{word}");
}}
"""
        elif difficulty == 2:
            x = rng.randint(1, 4)
            y = rng.randint(1, 5)
            target = x * y * rng.randint(1, 9) + rng.randint(1, 9)
            bfit_template = f"""
int main() {{
    int acc = 0;
    int target = {target};
    int x = {x};
    int y = {y};
    while (acc < target) {{
        acc = acc + x;
        acc = acc + y;
    }}
    printint(acc);
}}
"""
        elif difficulty == 3:
            x = rng.randint(1, 7)
            y = rng.randint(1, 9)
            target = x * y * rng.randint(1, 9) + rng.randint(1, 9) + 50
            conditional = target - rng.randint(1, 40)
            bfit_template = f"""
int main() {{
    int acc = 0;
    int target = {target};
    int x = {x};
    int y = {y};
    while (acc < target) {{
        acc = acc + x;
        if (acc > {conditional}) {{
            acc = acc + y;
        }}
    }}
    printint(acc);
}}
"""
        rendered_bfit = bfit_template
        return rendered_bfit

    def compile_bfit_code_to_bf(self, bfit: str) -> str:
        bf = Compiler.compile(bfit, optimize_code=True)
        # bf = Minify.minify(bf) # Is this necessary?
        return bf

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Determine if the solution provided solves the BF task.

        The function awards 1.0 for a correct answer.

        Args:
            answer (Optional[str]): The user's answer.
            entry (dict[str, Any]): The original dataset entry containing the correct answer.

        Returns:
            float: The computed score between 0.0 and 1.0.
        """

        if not isinstance(answer, str):
            return 0.0

        if answer == entry["answer"]:
            return 1.0  # Yay

        if entry["answer"] in answer.splitlines():
            # We can be quite confident that the correct answer was given
            # It was likely just given alongside an explanation
            return max(0.9 * len(answer) / len(entry["answer"]), 0.1)

        if entry["answer"] in answer:
            # Since answers are English words, some risk of the response coincidentally containing the answer
            return max(0.5 * len(answer) / len(entry["answer"]), 0.1)

        return 0.0


class BFCurriculum(BaseCurriculum):
    def __init__(self):
        super().__init__(BFCurriculum.__name__, BFConfig)

        # Define attributes
        self._define_attributes(
            ScalarAttributeDefinition(
                name="difficulty",
                field_name="difficulty",
                levels=[0, 1, 2, 3],
                description="Difficulty level",
            )
        )


# Register the dataset
register_dataset(DATASET_NAME, BFDataset, BFConfig, BFCurriculum)
