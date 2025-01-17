
from flask import Flask, render_template, request 
from transformers import AutoTokenizer, AutoModelForCausalLM 
import torch  
import multiprocessing  
import ast  
import traceback  


MODEL_NAME = "Salesforce/codegen-350M-mono"  


app = Flask(__name__)


def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


tokenizer, model = None, None

#  Generate Python Code
def generate_code(prompt, max_length=512):
    try:
        if not prompt.strip():
            return "Error: Prompt cannot be empty."

        if tokenizer is None or model is None:
            return "Error: Model failed to load."

        enriched_prompt = (
            f"Write a complete Python function for the following task:\n\n"
            f"{prompt}\n\nEnsure proper syntax, structure, and comments."
        )

        inputs = tokenizer.encode(enriched_prompt, return_tensors="pt", add_special_tokens=True)
        attention_mask = torch.ones(inputs.shape, device=inputs.device)

        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,
        )

        code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return code.strip()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return "Error: CUDA Out of Memory. Please reduce max_length or use a smaller model."
    except Exception as e:
        return f"Error: {e}"

#  Add Inline Comments
def add_inline_comments(code):
    """
    Add meaningful inline comments to Python code.
    """
    try:
        commented_lines = []
        for line in code.split('\n'):
            stripped_line = line.strip()
            if stripped_line.startswith('import'):
                commented_lines.append(f"{line}  # Import necessary libraries")
            elif stripped_line.startswith('def'):
                commented_lines.append(f"{line}  # Define a function")
            elif '=' in stripped_line and not stripped_line.startswith('#'):
                commented_lines.append(f"{line}  # Variable assignment")
            elif 'return' in stripped_line:
                commented_lines.append(f"{line}  # Return a value")
            elif stripped_line.startswith('if'):
                commented_lines.append(f"{line}  # Conditional statement")
            elif stripped_line.startswith('for') or stripped_line.startswith('while'):
                commented_lines.append(f"{line}  # Loop structure")
            elif stripped_line.startswith('print'):
                commented_lines.append(f"{line}  # Print output")
            else:
                commented_lines.append(line)  

        return '\n'.join(commented_lines)
    except Exception as e:
        return f"Error in adding comments: {e}"

#  Detect Errors in Python Code
def find_code_errors(code):
    """
    Analyze Python code for syntax and logical errors and provide solutions.
    """
    error_report = []

    # Check for syntax errors
    try:
        ast.parse(code)
    except SyntaxError as e:
        error_report.append(f"Syntax Error: {e.msg} at line {e.lineno}, column {e.offset}")
        error_report.append(f"Solution: Check syntax near line {e.lineno}, ensure proper indentation, punctuation, or keyword usage.")

    # Check for logical errors by executing the code
    try:
        exec_globals = {}
        exec(code, exec_globals)
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        tb = traceback.extract_tb(e.__traceback__)
        error_line = tb[-1].lineno if tb else 'Unknown'
        error_report.append(f"Logical Error: {error_type} at line {error_line}: {error_msg}")
        error_report.append(f"Solution: Review the logic around line {error_line}.")

    if not error_report:
        return "No errors found. The code appears to be valid and functional."
    
    return "\n".join(error_report)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_prompt = request.form.get("prompt", "").strip()
        action = request.form.get("action", "").strip()
        
        if action == "generate":
            generated_code = generate_code(user_prompt)
            return render_template("index.html", generated_code=generated_code)
        
        if action == "comment":
            commented_code = add_inline_comments(user_prompt)
            return render_template("index.html", commented_code=commented_code)
        
        if action == "error_check":
            error_report = find_code_errors(user_prompt)
            return render_template("index.html", generated_error=error_report)

    return render_template("index.html")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    tokenizer, model = load_model()
    if tokenizer and model:
        print("Model loaded successfully!")
    else:
        print("Failed to load the model. Please check the logs.")
    app.run(debug=True)
