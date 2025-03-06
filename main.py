import tkinter as tk
import json
import random
import re
import os

# Load aiMemory from local storage (aiData.json) or create default if not available
if os.path.exists("aiData.json"):
    with open("aiData.json", "r") as f:
        try:
            aiMemory = json.load(f)
        except Exception:
            aiMemory = {"formulas": {}, "classifications": {}}
else:
    aiMemory = {"formulas": {}, "classifications": {}}

# ANN1: 12 inputs, 15 hidden layers (12 neurons each), 12 outputs
class ANN1:
    def __init__(self):
        self.inputSize = 12
        self.hiddenLayers = 15
        self.hiddenSize = 12
        self.outputSize = 12
        self.weights = []
        self.biases = []

    def init(self):
        # Initialize weights and biases randomly
        for i in range(self.hiddenLayers):
            self.weights.append([random.random() for _ in [0] * self.hiddenSize])
            self.biases.append(random.random())
        self.weights.append([random.random() for _ in [0] * self.outputSize])
        self.biases.append(random.random())

    def train(self, input, output):
        # Simple training: store input-output pairs (for now)
        if not hasattr(self, 'trainingData'):
            self.trainingData = []
        self.trainingData.append({"input": input, "output": output})

    def predict(self, input):
        # Forward propagation
        activations = input
        for i in range(self.hiddenLayers):
            activations = [self.weights[i][j] * activations[j] + self.biases[i] for j in range(len(activations))]
            activations = [max(0, a) for a in activations]  # ReLU activation
        activations = [self.weights[self.hiddenLayers][j] * activations[j] + self.biases[self.hiddenLayers] for j in range(len(activations))]
        return activations

# ANN2: 2 inputs, 1 hidden layer (3 neurons), 2 outputs
class ANN2:
    def __init__(self):
        self.inputSize = 2
        self.hiddenLayers = 1
        self.hiddenSize = 3
        self.outputSize = 2
        self.weights = []
        self.biases = []

    def init(self):
        # Initialize weights and biases randomly
        for i in range(self.hiddenLayers):
            self.weights.append([random.random() for _ in [0] * self.hiddenSize])
            self.biases.append(random.random())
        self.weights.append([random.random() for _ in [0] * self.outputSize])
        self.biases.append(random.random())

    def train(self, input, output):
        # Simple training: store input-output pairs (for now)
        if not hasattr(self, 'trainingData'):
            self.trainingData = []
        self.trainingData.append({"input": input, "output": output})

    def predict(self, input):
        # Forward propagation
        activations = input
        for i in range(self.hiddenLayers):
            activations = [self.weights[i][j] * activations[j] + self.biases[i] for j in range(len(activations))]
            activations = [max(0, a) for a in activations]  # ReLU activation
        activations = [self.weights[self.hiddenLayers][j] * activations[j] + self.biases[self.hiddenLayers] for j in range(len(activations))]
        return activations

# Initialize ANNs
ann1 = ANN1()
ann2 = ANN2()
ann1.init()
ann2.init()

def processInput():
    inputText = inputData.get("1.0", tk.END).strip()
    if not inputText:
        return

    lines = inputText.split('\n')
    classification = {}
    predictions = []

    for input_line in lines:
        input_line = input_line.strip()
        if not input_line:
            continue
        # ANN Commands
        if input_line.startswith("/a "):
            temp = input_line.replace("/a ", "")
            if ":" in temp:
                training, output = [s.strip() for s in temp.split(":", 1)]
            else:
                training = temp
                output = ""
            training_list = [float(x) for x in training.split(",") if x.strip() != ""]
            output_list = [float(x) for x in output.split(",") if x.strip() != ""]
            if len(training_list) == 12 and len(output_list) == 12:
                ann1.train(training_list, output_list)
                predictions.append(f"ANN1 trained with input: {training_list} and output: {output_list}")
            elif len(training_list) == 2 and len(output_list) == 2:
                ann2.train(training_list, output_list)
                predictions.append(f"ANN2 trained with input: {training_list} and output: {output_list}")
            else:
                predictions.append("Invalid input/output size for training.")
        elif input_line.startswith("/t "):
            testing = [float(x) for x in input_line.replace("/t ", "").split(",") if x.strip() != ""]
            if len(testing) == 12:
                result = ann1.predict(testing)
                predictions.append(f"ANN1 test result: {result}")
            elif len(testing) == 2:
                result = ann2.predict(testing)
                predictions.append(f"ANN2 test result: {result}")
            else:
                predictions.append("Invalid input size for testing.")
        elif input_line.startswith("/dl"):
            ann1.trainingData = []
            ann2.trainingData = []
            predictions.append("ANN1 and ANN2 data cleared.")
        # Process for Question Handling
        if input_line.endswith("?"):
            question = input_line.replace("?", "").strip()
            if question.startswith("{"):
                # Handle similarity, dissimilarity, or reconstruction
                if "}:{" in question:
                    parts = question.split("}:{")
                    obj1 = parts[0].replace("{", "").replace("}", "").strip()
                    obj2 = parts[1].replace("{", "").replace("}", "").strip()
                    if obj2.startswith("!"):
                        # Dissimilarity detection
                        obj2 = obj2.replace("!", "").strip()
                        result = findDissimilarities(obj1, obj2)
                        predictions.append(f"Dissimilarities between {obj1} and {obj2}:\n{json.dumps(result, indent=2)}")
                    else:
                        # Similarity detection
                        result = findSimilarities(obj1, obj2)
                        predictions.append(f"Similarities between {obj1} and {obj2}:\n{json.dumps(result, indent=2)}")
                elif " - " in question:
                    # Reconstruction
                    components, target = [s.strip() for s in question.split(" - ", 1)]
                    result = reconstructObject(components, target)
                    predictions.append(f"Reconstructed {target}:\n{json.dumps(result, indent=2)}")
            elif question.startswith("["):
                # Unsupervised learning for array-like input
                items = [s.strip() for s in question.replace("[", "").replace("]", "").split(",") if s.strip() != ""]
                result = unsupervisedLearning(items)
                predictions.append(f"Unsupervised Learning:\n{json.dumps(result, indent=2)}")
            else:
                predictions.append(processQuestion(question))
        elif input_line.startswith("del "):
            obj = input_line.replace("del ", "").strip()
            deleteObject(obj)
            predictions.append(f"Deleted {obj} and all its properties.")
        elif "=" in input_line and "->" not in input_line:
            key, formula = [s.strip() for s in input_line.split("=", 1)]
            value = evaluateFormula(formula)
            aiMemory[key] = value
            predictions.append(f"{key} = {value}")
        elif "->" in input_line:
            parts = [s.strip() for s in input_line.split("->")]
            obj = aiMemory
            for i in range(len(parts) - 1):
                if parts[i] not in obj:
                    obj[parts[i]] = {}
                obj = obj[parts[i]]
            last_part = parts[-1]
            if ":" in last_part:
                lastKey, value = [s.strip() for s in last_part.split(":", 1)]
            else:
                lastKey = last_part
                value = ""
            obj[lastKey] = value
            classification = aiMemory
        elif input_line.startswith("["):
            # Auto-classification or unsupervised learning
            if "]" in input_line:
                items_part, sep, categories_part = input_line.partition("]")
                items = [s.strip() for s in items_part.replace("[", "").split(",") if s.strip() != ""]
                categories = categories_part.strip()
                if categories:
                    categories = [s.strip() for s in categories.split(" ") if s.strip() != ""]
                    result = autoClassify(items, categories)
                    predictions.append(f"Auto-Classification:\n{json.dumps(result, indent=2)}")
                else:
                    result = unsupervisedLearning(items)
                    predictions.append(f"Unsupervised Learning:\n{json.dumps(result, indent=2)}")
        # End of per-line processing

    with open("aiData.json", "w") as f:
        json.dump(aiMemory, f)
    classificationOutput.config(state=tk.NORMAL)
    classificationOutput.delete("1.0", tk.END)
    classificationOutput.insert(tk.END, json.dumps(classification, indent=2))
    classificationOutput.config(state=tk.DISABLED)
    predictionOutput.config(state=tk.NORMAL)
    predictionOutput.delete("1.0", tk.END)
    if predictions:
        predictionOutput.insert(tk.END, "\n".join(predictions))
    else:
        predictionOutput.insert(tk.END, "No prediction.")
    predictionOutput.config(state=tk.DISABLED)

def evaluateFormula(formula):
    vars = list(aiMemory.keys())
    for v in vars:
        regex = re.compile(r'\b' + re.escape(v) + r'\b')
        # If the value is not a string, convert to string
        formula = regex.sub(str(aiMemory[v]), formula)
    try:
        return eval(formula)
    except Exception:
        return "Invalid Formula"

# Process Question
def processQuestion(question):
    questionParts = question.split(" ")
    if len(questionParts) >= 2 and questionParts[0] == "formula":
        # Handle formula questions like "formula a apple=2 banana=3"
        formulaName = questionParts[1]
        if "formulas" in aiMemory and formulaName in aiMemory["formulas"]:
            formula = aiMemory["formulas"][formulaName]
            variables = {}
            for i in range(2, len(questionParts)):
                if "=" in questionParts[i]:
                    varName, value = [s.strip() for s in questionParts[i].split("=", 1)]
                    variables[varName] = float(value)
            result = evaluateFormulaWithVariables(formula, variables)
            return f"Result of formula {formulaName} = {result}"
        else:
            return f"Formula {formulaName} not found."
    elif len(questionParts) == 2:
        entity = questionParts[0]
        property = questionParts[1]
        # Handle specific questions like "USA currency"
        if entity in aiMemory and isinstance(aiMemory[entity], dict) and property in aiMemory[entity]:
            return f"{property} of {entity} = {aiMemory[entity][property]}"
        else:
            return f"I couldn't find {property} for {entity}."
    elif len(questionParts) == 1:
        entity = questionParts[0]
        # Handle general questions like "USA?"
        if entity in aiMemory and isinstance(aiMemory[entity], dict):
            result = []
            for key in aiMemory[entity]:
                result.append(f"{key} of {entity} = {aiMemory[entity][key]}")
            return "\n".join(result)
        else:
            return f"I couldn't find any information about {entity}."
    return "I couldn't find an answer to your question."

def evaluateFormulaWithVariables(formula, variables):
    vars_keys = list(variables.keys())
    for v in vars_keys:
        regex = re.compile(r'\b' + re.escape(v) + r'\b')
        formula = regex.sub(str(variables[v]), formula)
    try:
        return eval(formula)
    except Exception:
        return "Invalid Formula"

# Auto-Classification
def autoClassify(items, categories):
    result = {}
    for category in categories:
        result[category] = []
    for item in items:
        bestMatch = {"category": None, "similarity": -1}
        for category in categories:
            similarity = stringSimilarity(item, category)
            if similarity > bestMatch["similarity"]:
                bestMatch["category"] = category
                bestMatch["similarity"] = similarity
        if bestMatch["category"]:
            result[bestMatch["category"]].append(item)
    return result

# Unsupervised Learning
def unsupervisedLearning(items):
    clusters = {}
    for item in items:
        foundCluster = False
        for cluster in clusters:
            if stringSimilarity(item, cluster) > 0.6:  # Threshold for similarity
                clusters[cluster].append(item)
                foundCluster = True
                break
        if not foundCluster:
            clusters[item] = [item]
    return clusters

# String Similarity (Levenshtein Distance)
def stringSimilarity(a, b):
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    matrix = []
    for i in range(len(b) + 1):
        matrix.append([i])
    for j in range(len(a) + 1):
        matrix[0].append(j)
    for i in range(1, len(b) + 1):
        for j in range(1, len(a) + 1):
            if b[i - 1] == a[j - 1]:
                matrix[i].append(matrix[i - 1][j - 1])
            else:
                matrix[i].append(min(
                    matrix[i - 1][j - 1] + 1,  # Substitution
                    matrix[i][j - 1] + 1,      # Insertion
                    matrix[i - 1][j] + 1       # Deletion
                ))
    return 1 - (matrix[len(b)][len(a)] / max(len(a), len(b)))

# Find Similarities
def findSimilarities(obj1, obj2):
    result = {}
    if obj1 in aiMemory and obj2 in aiMemory:
        for key in aiMemory[obj1]:
            if key in aiMemory[obj2] and aiMemory[obj1][key] == aiMemory[obj2][key]:
                result[key] = aiMemory[obj1][key]
        # Store similarities in a new class
        simClass = f"{obj1}_sim_{obj2}"
        aiMemory[simClass] = result
    return result

# Find Dissimilarities
def findDissimilarities(obj1, obj2):
    result = {obj1: {}, obj2: {}}
    if obj1 in aiMemory and obj2 in aiMemory:
        for key in aiMemory[obj1]:
            if key not in aiMemory[obj2]:
                result[obj1][key] = aiMemory[obj1][key]
        for key in aiMemory[obj2]:
            if key not in aiMemory[obj1]:
                result[obj2][key] = aiMemory[obj2][key]
        # Store dissimilarities in a new class
        disClass = f"{obj1}_dis_{obj2}"
        aiMemory[disClass] = result
    return result

# Delete Object
def deleteObject(obj):
    if obj in aiMemory:
        del aiMemory[obj]
        # Remove any similarity or dissimilarity classes involving the object
        keys_to_delete = []
        for key in list(aiMemory.keys()):
            if f"_sim_{obj}" in key or f"_dis_{obj}" in key:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del aiMemory[key]

# Reconstruct Object
def reconstructObject(components, target):
    result = {}
    # Split components by "}{"
    comps = components.split("}{")
    for comp in comps:
        comp = comp.replace("{", "").replace("}", "").strip()
        if comp in aiMemory:
            for key in aiMemory[comp]:
                result[key] = aiMemory[comp][key]
    aiMemory[target] = result
    return result

# Set up tkinter UI
root = tk.Tk()
root.title("Arlix Pocket AI")

# Input field
inputLabel = tk.Label(root, text="Input Data:")
inputLabel.pack()
inputData = tk.Text(root, height=10, width=80)
inputData.pack()

# Button to process input
processButton = tk.Button(root, text="Process", command=processInput)
processButton.pack()

# Classification output
classificationLabel = tk.Label(root, text="Classification Output:")
classificationLabel.pack()
classificationOutput = tk.Text(root, height=10, width=80, state=tk.DISABLED)
classificationOutput.pack()

# Prediction output
predictionLabel = tk.Label(root, text="Prediction Output:")
predictionLabel.pack()
predictionOutput = tk.Text(root, height=10, width=80, state=tk.DISABLED)
predictionOutput.pack()

root.mainloop()
