from datasets import load_dataset

# Load the C++ split from the code-to-text subset of CodeXGLUE
cpp_dataset = load_dataset("microsoft/codexglue_code_to_text", "cpp", split="train", streaming=True)

# You can now access the data
print("--- Displaying the first 10 examples ---")
for i, example in enumerate(cpp_dataset):
    print(f"\n--- Example {i+1} ---")
    print("DOCSTRING:", example['docstring'])
    print("CODE:", example['code'])

    # Stop the loop after the 10th example (index 9)
    if i >= 9:
        break