from flask import Flask
import telegram
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ConversationHandler,
    ContextTypes,
)
import os
from dotenv import load_dotenv
import logging
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from mem0 import MemoryClient

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
app = Flask(__name__)
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "your_bot_token_here")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MEM0_API_KEY = os.getenv("MEM0_API_KEY")

# Initialize LangChain LLM
langchain_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    api_key=GEMINI_API_KEY
)

# Initialize Mem0 client
mem0_client = MemoryClient(api_key=MEM0_API_KEY)

# Bot states
ASSISTANT_TYPE, LANGUAGE, QUERY = range(3)

# In-memory user state storage
user_data = {}

# Session histories
legal_history = InMemoryChatMessageHistory()
veterinary_history = InMemoryChatMessageHistory()
financial_history = InMemoryChatMessageHistory()

def get_session_history(assistant_type: str):
    MAX_MESSAGES = 5
    histories = {
        'legal': legal_history,
        'veterinary': veterinary_history,
        'financial': financial_history
    }
    history = histories.get(assistant_type, legal_history)
    if len(history.messages) > MAX_MESSAGES:
        history.messages = history.messages[-MAX_MESSAGES:]
    return history

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start the conversation and ask for assistant type."""
    user_id = update.message.from_user.id
    user_data[user_id] = {}  # Initialize user data
    logger.debug(f"User {user_id} started conversation")
    await update.message.reply_text(
        "Welcome to Krishi.ai Assistant Bot! 🌾\nWhich assistant would you like to use?",
        reply_markup=ReplyKeyboardMarkup(
            [["Financial Assistant"], ["Legal Assistant"], ["Veterinary Assistant"]],
            one_time_keyboard=True
        ),
    )
    return ASSISTANT_TYPE

async def get_assistant_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle assistant type selection."""
    user_id = update.message.from_user.id
    text = update.message.text.strip()
    logger.debug(f"User {user_id} selected assistant: {text}")
    assistant_map = {
        "Financial Assistant": "financial",
        "Legal Assistant": "legal",
        "Veterinary Assistant": "veterinary"
    }
    if text not in assistant_map:
        logger.warning(f"User {user_id} invalid assistant: {text}")
        await update.message.reply_text(
            "Please select a valid assistant:",
            reply_markup=ReplyKeyboardMarkup(
                [["Financial Assistant"], ["Legal Assistant"], ["Veterinary Assistant"]],
                one_time_keyboard=True
            ),
        )
        return ASSISTANT_TYPE
    user_data[user_id]["assistant_type"] = assistant_map[text]
    logger.debug(f"User {user_id} assistant_type set to {user_data[user_id]['assistant_type']}")
    await update.message.reply_text(
        "Please select the language:",
        reply_markup=ReplyKeyboardMarkup(
            [["English"], ["Hindi"], ["Kannada"]],
            one_time_keyboard=True
        ),
    )
    return LANGUAGE

async def get_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle language selection."""
    user_id = update.message.from_user.id
    text = update.message.text.strip()
    logger.debug(f"User {user_id} selected language: {text}")
    language_map = {
        "English": "en",
        "Hindi": "hi",
        "Kannada": "kn"
    }
    if text not in language_map:
        logger.warning(f"User {user_id} invalid language: {text}")
        await update.message.reply_text(
            "Please select a valid language:",
            reply_markup=ReplyKeyboardMarkup(
                [["English"], ["Hindi"], ["Kannada"]],
                one_time_keyboard=True
            ),
        )
        return LANGUAGE
    user_data[user_id]["language"] = language_map[text]
    logger.debug(f"User {user_id} language set to {user_data[user_id]['language']}")
    assistant_type = user_data[user_id]["assistant_type"]
    assistant_names = {
        "financial": "Financial Assistant",
        "legal": "Legal Assistant",
        "veterinary": "Veterinary Assistant"
    }
    await update.message.reply_text(
        f"Ask your question to the {assistant_names[assistant_type]} (in {text}):",
        reply_markup=ReplyKeyboardRemove(),
    )
    return QUERY

async def get_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle user query and generate response."""
    user_id = update.message.from_user.id
    query = update.message.text.strip()
    logger.debug(f"User {user_id} query: {query}")
    if not query:
        logger.warning(f"User {user_id} empty query")
        await update.message.reply_text("Please enter a valid question:")
        return QUERY

    if user_id not in user_data or "assistant_type" not in user_data[user_id] or "language" not in user_data[user_id]:
        logger.error(f"User {user_id} missing user_data, restarting")
        await update.message.reply_text(
            "Session expired. Please start again with /start.",
            reply_markup=ReplyKeyboardRemove()
        )
        return ConversationHandler.END

    data = user_data[user_id]
    assistant_type = data["assistant_type"]
    language = data["language"]

    # Show typing action
    await context.bot.send_chat_action(
        chat_id=update.message.chat.id, action=telegram.constants.ChatAction.TYPING
    )

    # Prepare prompt based on assistant type
    language_names = {"en": "English", "hi": "Hindi", "kn": "Kannada"}
    language_name = language_names[language]

    prompt_templates = {
        "financial": """ 
        You are a friendly financial assistant for rural Indian farmers. Provide a clear and concise answer to the following question related to agricultural loans, subsidies, or financial schemes. The answer must be in {language_name} and tailored to the Indian context (e.g., referencing Indian banks, government schemes, or local financial institutions). Limit the response to 3-5 sentences for brevity. Use simple language suitable for farmers and avoid financial jargon. If the question is too vague or unrelated to agricultural finance, return a polite message indicating the need for a more specific financial question.

        Question: {query}

        Instructions:
        - Answer in {language_name}, using simple and clear language.
        - Focus on practical advice or information relevant to agricultural loans, subsidies, or financial schemes in India.
        - If unsure, suggest consulting a local bank or government agriculture office.
        - Do not include markdown, code fences, or additional text—only the plain text response.
        - Do not use any special symbols like *
        - If user is greeting, then you also greet
        """,
        "legal": """ 
        You are a friendly legal assistant for rural Indian farmers. Provide a clear and concise answer to the following legal question related to agricultural laws, land disputes, or government schemes. The answer must be in {language_name} and tailored to the Indian context (e.g., referencing Indian laws, government schemes, or local authorities). Limit the response to 3-5 sentences for brevity. Use simple language suitable for farmers and avoid legal jargon. If the question is too vague or unrelated to agricultural legal issues, return a polite message indicating the need for a more specific legal question.

        Question: {query}

        Instructions:
        - Answer in {language_name}, using simple and clear language.
        - Focus on practical advice or information relevant to agricultural laws, land disputes, or government schemes in India.
        - If unsure, suggest consulting a local lawyer or government office.
        - Do not include markdown, code fences, or additional text—only the plain text response.
        - Do not use any special symbols like *
        - If user is greeting, then you also greet
        """,
        "veterinary": """ 
        You are a friendly veterinary assistant for rural Indian farmers. Provide a clear and concise answer to the following question related to livestock health, animal diseases, or veterinary care. The answer must be in {language_name} and tailored to the Indian context (e.g., referencing common Indian livestock, local remedies, or veterinary services). Limit the response to 3-5 sentences for brevity. Use simple language suitable for farmers and avoid technical jargon. If the question is too vague or unrelated to livestock health, return a polite message indicating the need for a more specific question.

        Question: {query}

        Instructions:
        - Answer in {language_name}, using simple and clear language.
        - Focus on practical advice or information relevant to livestock health, animal diseases, or veterinary care in India.
        - If unsure, suggest consulting a local veterinarian or government veterinary office.
        - Do not include markdown, code fences, or additional text—only the plain text response.
        - Do not use any special symbols like *
        - If user is greeting, then you also greet
        """
    }

    prompt_template = PromptTemplate(
        input_variables=["query", "language_name"],
        template=prompt_templates[assistant_type]
    )

    prompt = prompt_template.format(query=query, language_name=language_name)

    try:
        logger.debug(f"User {user_id} sending prompt to Gemini")
        response = langchain_llm.invoke(prompt)
        logger.debug(f"User {user_id} Gemini response: {response.content}")

        answer = response.content.strip()
        if not answer:
            logger.warning(f"User {user_id} empty Gemini response")
            answer = "No answer found. Please ask a more specific question."

        # Store in Mem0
        mem0_client.add(
            messages=[
                {"role": "user", "content": query},
                {"role": "assistant", "content": answer}
            ],
            user_id="aryan",
            output_format="v1.1"
        )
        logger.debug(f"User {user_id} stored in Mem0")

        # Add to session history
        session_history = get_session_history(assistant_type)
        session_history.add_user_message(query)
        session_history.add_ai_message(answer)
        logger.debug(f"User {user_id} updated session history for {assistant_type}")

        await update.message.reply_text(
            answer,
            reply_markup=ReplyKeyboardRemove()
        )
        return QUERY


    except Exception as e:
        logger.error(f"User {user_id} Gemini query error: {str(e)}")
        await update.message.reply_text(
            "Error processing query. Please try again or start over with /start.",
            reply_markup=ReplyKeyboardRemove()
        )
        return ConversationHandler.END

async def continue_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle continuation option."""
    user_id = update.message.from_user.id
    text = update.message.text.strip()
    logger.debug(f"User {user_id} continuation option: {text}")

    if user_id not in user_data:
        logger.error(f"User {user_id} missing user_data, restarting")
        await update.message.reply_text(
            "Session expired. Please start again with /start.",
            reply_markup=ReplyKeyboardRemove()
        )
        return ConversationHandler.END

    if text == "Ask Another Question":
        assistant_type = user_data[user_id]["assistant_type"]
        language = user_data[user_id]["language"]
        language_names = {"en": "English", "hi": "Hindi", "kn": "Kannada"}
        assistant_names = {
            "financial": "Financial Assistant",
            "legal": "Legal Assistant",
            "veterinary": "Veterinary Assistant"
        }
        logger.debug(f"User {user_id} asking another question with {assistant_type} in {language}")
        await update.message.reply_text(
            f"Ask another question to the {assistant_names[assistant_type]} (in {language_names[language]}):",
            reply_markup=ReplyKeyboardRemove(),
        )
        return QUERY
    else:
        logger.warning(f"User {user_id} invalid continuation option: {text}")
        await update.message.reply_text(
            "Please select an option:",
            reply_markup=ReplyKeyboardMarkup(
                [["Ask Another Question"]],
                one_time_keyboard=True
            ),
        )
        return QUERY

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel the conversation."""
    user_id = update.message.from_user.id
    logger.debug(f"User {user_id} cancelled conversation")
    user_data.pop(user_id, None)
    await update.message.reply_text(
        "Operation cancelled. Start again with /start.",
        reply_markup=ReplyKeyboardRemove()
    )
    return ConversationHandler.END

# Initialize Telegram bot application
application = (
    Application.builder()
    .token(BOT_TOKEN)
    .build()
)

# Setup conversation handler
conv_handler = ConversationHandler(
    entry_points=[CommandHandler("start", start)],
    states={
        ASSISTANT_TYPE: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_assistant_type)],
        LANGUAGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_language)],
        QUERY: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_query)],
    },
    fallbacks=[
        CommandHandler("cancel", cancel),
        MessageHandler(filters.Regex(r"^(Ask Another Question)$"), continue_conversation),
    ],
)

application.add_handler(conv_handler)

if __name__ == "__main__":
    logger.info("Starting bot with polling")
    # Start polling
    application.run_polling(allowed_updates=Update.ALL_TYPES)