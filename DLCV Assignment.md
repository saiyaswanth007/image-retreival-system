You are a rigorous computer vision researcher. Your goal is to solve **Task 1: Classical Image Retrieval using LBP** with maximum correctness, clarity, and reproducibility.

Follow these rules strictly:

- Do NOT assume results or invent performance numbers
    
- Every step must be implementable in code
    
- Use only well-defined operations (no vague descriptions)
    
- Stay strictly within Task 1 scope
    

---

TASK OBJECTIVE (FROM ASSIGNMENT)

Implement a classical image retrieval system using Local Binary Patterns (LBP):

fi = LBP(Ii)

You must:

- Extract LBP features
    
- Compute similarity between images
    
- Retrieve Top-K similar images
    
- Analyze strengths and limitations
    

---

STEP 1: DEFINE LBP MATHEMATICALLY

Clearly define:

- Neighborhood (e.g., 3×3)
    
- Binary encoding rule:  
    Compare each neighbor pixel with center pixel
    
- Binary to decimal conversion
    

Provide the exact formula for LBP at a pixel.

---

STEP 2: FEATURE REPRESENTATION

Define how image-level feature is constructed:

- Histogram of LBP codes
    
- Number of bins
    
- Normalization method
    

Explain why histogram is used.

---

STEP 3: SIMILARITY METRIC

Define distance function explicitly:

- Euclidean OR Chi-square OR Cosine
    

Write the exact formula.

Explain:

- What property of LBP this metric captures
    

---

STEP 4: RETRIEVAL PIPELINE

Describe step-by-step pipeline:

1. Input query image
    
2. Compute LBP feature
    
3. Compare with all dataset features
    
4. Rank images by distance
    
5. Return Top-K
    

Make it algorithmic (clear steps, no ambiguity)

---

STEP 5: COMPUTATIONAL COMPLEXITY

Estimate:

- Feature extraction cost per image
    
- Retrieval cost (linear scan)
    

---

STEP 6: FAILURE MODE ANALYSIS (CRITICAL)

Explain WHY LBP fails:

- No semantic understanding
    
- Sensitive to noise
    
- Cannot capture global structure
    
- High-dimensional histogram issues
    

Each point must be explained mechanistically (not vague claims)

---

STEP 7: WHEN LBP WORKS

Clearly state scenarios where LBP is effective:

- Texture-dominant images
    
- Uniform backgrounds
    
- Low intra-class variation
    

---

STEP 8: SELF-CHECK

Before answering, verify:

- Are all formulas correct and standard?
    
- Is every step implementable?
    
- Are there any unsupported claims?
    

If yes → fix before output.

---

OUTPUT FORMAT:

- Section-wise structured answer
    
- Include formulas
    
- No unnecessary theory
    
- No fabricated results
    
- Focus on correctness + clarity