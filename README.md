# Prompt Engineering Analysis

This repository contains data, scripts, and results for a comprehensive study on **Prompt Engineering** across three main dimensions:

- **Patents** – Lens.org dataset and topic modeling  
- **Job Postings** – skills and techniques extracted, clustered, and analyzed  
- **University Courses** – syllabi analyzed and mapped to SWEBOK  

The goal is to **triangulate industry demand (patents, jobs) with academic supply (courses)** to surface competency gaps and opportunities in software engineering education.

---

## 📂 Repository Structure

### 1. Patents on Prompt Engineering
- **Patents Data downloaded from Lens.org (Excel)** – Raw dataset of patents retrieved  
- **TOPIC MODELING 8 TOPICS (PNG)** – Visualization of latent topics identified  
- **Topic Modeling Procedure (LDA) (Word)** – Documentation of the LDA pipeline  
- **TOPIC MODELING Video (MP4)** – Video walkthrough of topic modeling process  

---

### 2. Job Postings Analysis
- **Raw Data / Cleaned Job Descriptions (Excel)** – Original and cleaned datasets of job postings  
- **Analysis Per Skills (Excel)** – Breakdown of technical and prompting skills  
- **Analysis of Prompt Techniques coverage (Excel)** – Coverage of specific prompting techniques  
- **Clustering Output (TXT)** – Results from K-means clustering  
- **Clustering Python Code (TXT)** – Script used for clustering  
- **promptEngineeringTechniques_kmeans (Excel)** – Mapped clustering results  
- **Python Code (Data Cleaning and Extraction) (TXT)** – Preprocessing scripts  
- **PNG Visualizations** – Cluster compositions, task distributions, and skills taxonomy  

---

### 3. Courses Analysis
- **Courses Data (cleaned, enriched) / Courses Data Final (Excel)** – Extracted and curated data from university courses  
- **course analysis pipeline.py (Python)** – Script to automate cleaning and enrichment  
- **List of Universities (Word)** – Institutions included in the dataset  
- **Prompt to Extract Data per Course (Word)** – Prompting method used to collect course data  
- **swebok_mapping (Excel)** – Mapping of course content to SWEBOK categories  
- **taxonomy.json** – Structured taxonomy of skills and techniques  
- **Clustering and Co-occurrence Analysis (Folder)** – Results of clustering and skill co-occurrence patterns  

---

## 🔍 Methods Overview
- **Topic Modeling (LDA):** Applied to patent abstracts to identify emerging areas  
- **Clustering (K-means):** Used on job descriptions to detect demand patterns  
- **Taxonomy Development:** Prompt engineering skills and techniques organized into structured categories  
- **SWEBOK Mapping:** Courses aligned with software engineering knowledge areas  

---

## 📊 Outputs
- Topic models of patents (figures, documentation, video)  
- Cluster analyses of job postings (Excel + PNG)  
- Course coverage analysis (Excel + SWEBOK alignment)  
- JSON taxonomy for skills and techniques  

---

## 🚀 How to Use
1. **Patents** – Review `Patents Data downloaded from Lens.org` and refer to `Topic Modeling Procedure (LDA)` for replication  
2. **Jobs** – Start with `Cleaned Job Descriptions`, then explore `Clustering Output` and visualizations  
3. **Courses** – Use `Courses Data Final` and `swebok_mapping` for curriculum alignment studies  
4. **Scripts** – Python/TXT scripts are provided for cleaning, clustering, and pipeline automation  

---

## 📌 Notes
- **Data sources:** Lens.org (patents), job postings extracted from LinkedIn (June–July 2025) from all countries ranked among the top 50 economies in the WIPO Global'24, and university course catalogs  
- Ensure proper environment setup for Python scripts (`scikit-learn`, `pandas`, `matplotlib`)  
- Large artifacts (video, PNGs) illustrate intermediate and final results  
