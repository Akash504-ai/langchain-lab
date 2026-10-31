# Blog Writter / Generator - Bot

from dotenv import load_dotenv
load_dotenv()

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser