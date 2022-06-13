import streamlit.components.v1 as components
import streamlit as st

with st.echo(code_location="below"):
    st.write("## Telegram-Bot")

    st.write("I created a Telegram-Bot which finds the best elementary school in a given city, and provides information about the average house price in the city.")

    st.write("I uploaded all my files in the form ")
    st.write("In order to determine if text input was not in a correct form, I used regex.")

    st.write("Also, I use SQL  for making fast resquests and getting the necessary information about the housing prices and schools."
             "See the code below.")
    st.write("You can access my bot any time: @CaliforniaLoveBot.")

    HtmlFile = open('Telegram-Bot.html', 'r', encoding='utf-8')

    components.html(HtmlFile.read(), height=500, width=1000, scrolling=True)

    import streamlit.components.v1 as components

    #from telegram import ForceReply
    #from telegram import Update
    #from telegram.ext import MessageHandler
    #from telegram.ext import filters
    #from telegram.ext import Application
    #from telegram.ext import ContextTypes, CommandHandler
    #import sqlite3
    #import logging
    #import re

    #TOKEN = '...' (confidential!)

    #logging.basicConfig(
    #    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
    #)
    #logger = logging.getLogger(__name__)

    #async def start(update: Update, context: ContextTypes) -> None:
    #    user = update.effective_user
    #    await update.message.reply_html(
    #        rf"Hi {user.mention_html()}! This bot is created to help you find the best elementary school in your city! For example, try writing 'San Francisco'",
    #        reply_markup=ForceReply(selective=True),
    #    )

    #async def elemschool(update: Update, context: ContextTypes) -> None:
    #    if re.fullmatch(r'^([A-Z][a-z]* *)+$', update.message.text):
    #        conn = sqlite3.connect("school_n_price.sqlite")
    #        c = conn.cursor()
    #        price = c.execute('SELECT list_price FROM school_n_price WHERE City = ?', [f'{update.message.text}']).fetchall()
    #        school = c.execute('SELECT Name FROM school_n_price WHERE City = ?', [f'{update.message.text}']).fetchall()
    #        rate = c.execute('SELECT Rating FROM school_n_price WHERE City = ?', [f'{update.message.text}']).fetchall()
    #        c.close()
    #        await update.message.reply_text(f"In your town the best elementary school is {school[0][0]} with rating {rate[0][0]}/10 and median house value is ${str(int(float(price[0][0])))}")
    #    else:
    #        await update.message.reply_text("Wrong format. Please make sure that all words in your city name are capitalized.")


    #def main():
    #    application = Application.builder().token(TOKEN).build()
    #    application.add_handler(CommandHandler("start", start))
    #   application.add_handler(CommandHandler("help", start))
    #    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, elemschool))
    #    application.run_polling()