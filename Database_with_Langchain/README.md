## Description

This project provides a simple chat interface to interact with a MySQL database. Users can ask questions, and the system generates and executes SQL queries based on the user's input using LangChain and Google Generative AI. The query results are then displayed to the user through a Streamlit application.


## Obtaining an API key

We used an API key obtained via https://ai.google.dev/gemini-api/docs/api-key.


## Database

MySQL Workbench was used as the primary tool for managing and interacting with the MySQL database.

MySQL Workbench is a comprehensive solution that offers database design, data management, and SQL query execution functionalities. We used it to create, modify, and manage the tables of the "retail_sales_db" database, as well as to test and optimize SQL queries before integrating them into the application.

## Improving model performance with Few-Shot learning
To improve the accuracy and performance of the model in generating complex SQL queries, we used few-shot learning. This approach involves training the model with a limited set of representative examples of natural language questions and their associated SQL queries. By providing the model with varied and specific examples, it learns to generate more accurate and contextually appropriate SQL queries based on the database structure. Additionally, by clearly defining the database schema (tables and columns), we ensure that the generated queries are relevant and well-formed. This method optimizes the model's ability to handle complex queries, improving user experience and result accuracy.

