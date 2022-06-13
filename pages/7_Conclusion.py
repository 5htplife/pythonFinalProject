import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np


with st.echo(code_location="below"):
    st.write("## Conclusion")
    st.write("I hope you enjoyed the project and got some insights about the housing prices in California.")
    st.write("Obviously, California is not a cheap place to live in.")

    @st.cache(allow_output_mutation=True)
    def get_main_data():
        return pd.read_csv("https://github.com/5htplife/FinalProject/raw/main/house_prices_california2.csv")

    main_data = get_main_data()

    salary = st.number_input(label = "Insert your (planned) annual salary", value = 1, key="number", step=1)


    def calculate_number_of_years(salary):
        saved = 0.2 * int(salary)
        average = main_data['list_price'].mean()
        return average / saved

    a = calculate_number_of_years(salary)

    if salary:
        a
        if a > 10:
            st.write("Wow, that's a lot!")
        else:
            st.write("Not to long, though.")

    st.write("At least, you get this:")

    image = Image.open('cool_house.webp')

    st.image(image, caption='A four-bedroom Victorian house in Redlands ($2 mln)')

    st.write("Well, maybe not always...")
    st.write("For the same amount you may get this:")

    image2 = Image.open('worst_house.jpeg')

    st.image(image2, caption='House in San Francisco (sold $2 mln)')
