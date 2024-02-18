---
layout: project
type: project
image: img/projects/MOD/MOD_logo.png
title: "Recipts Parser using ChatGPT"
date: 2022-05-10
published: true
labels:
  - OpenAI
  - Pydantic
  - Python
  - JSON
  - Machine Learning
  - Github
  - Optical Character Recognition
  - FAISS
summary: "Working with three other of my ICS 438 classmates, we constructed a program which would parse the raw text obtained from using optical character recognition (OCR) on random consumer receipts into JSON objects using OpenAI's large language model, ChatGPT. It would then provide analytics on the JSON data and use it to train a KNN model from the FAISS library (open-sourced from Facebook AI) to classify new receipts into categories."
---

<h2>Links</h2>
View the data visualization [here](https://receipt-classification-visualization.streamlit.app/).

View the [Organization GitHub page](https://github.com/manoa-organization-database).

View the parse, conversion, and classification [source code](https://github.com/RecieptsParse/OCR_TO_JSON).

View the visualization [source code](https://github.com/RecieptsParse/visualization).
