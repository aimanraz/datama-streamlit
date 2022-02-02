
# import necessary packages
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

st.set_option('deprecation.showPyplotGlobalUse', False)
sns.set_theme(style='dark',palette="deep")

# set the page title
st.set_page_config(page_title='Datama')
# import banner image
image = Image.open("Banner.png")
st.image(image)

# function
def about_data(data):
	#display top 5 row
	with st.expander("Top 5 Rows"):
		st.dataframe(data.head())

	#display the shape of dataset
	with st.expander('How big is the dataset?'):
		st.write(f"The dataset contains {data.shape[0]} rows with {data.shape[1]} columns.")

	#display the data types foe each columns
	with st.expander('What is the type of each column?'):
		col = st.columns(2)
		for col_name,col_type in zip(data.dtypes.index,data.dtypes):
			if col_type != 'object':
				col_type = 'Numerical'
			else:
				col_type = 'Categorical'
			col[0].write(col_name)
			col[1].write(col_type)

	# display the nulls value in each columns
	with st.expander("Checking nulls in each columns"):
		df_null = pd.DataFrame(data.isnull().sum(), columns=['Total nulls'])
		st.dataframe(df_null)

	# cheking number of duplicated rows
	with st.expander("Is there any duplicated records?"):
		num_dup = data.duplicated().sum()
		if num_dup != 0:
			st.warning(f'There is {num_dup} duplicated rows.')
		else:
			st.info('No duplicated record detected')

	# uniques value in each column
	with st.expander("Uniques in each columns"):
		df_uniq = pd.DataFrame(data.nunique(), columns=['Total'])
		st.dataframe(df_uniq)

	with st.expander("Basic statistical summary"):
		st.dataframe(data.describe())

def viz(data):

	analysis_type = st.radio("Choose one type of statistical analyis", ('Univariate','Bivariate','Multivariate'))
	if analysis_type == 'Univariate':
		attr = st.selectbox('Choose your attributes', data.columns)

		if (data[attr].dtypes == 'object'):

			if len(data[attr].unique()) < 15:

				with st.expander("Pie Chart"):
					fig = plt.figure()
					title = "Proportion of "+str(attr)
					plt.title(title, y=1.03, fontsize=15, weight='bold')
					plt.pie(data[attr].value_counts(normalize=True), labels=data[attr].unique(),autopct='%0.f%%')
					st.pyplot(fig)

				with st.expander("Countplot"):
					fig = plt.figure()
					title = "Comparison of "+str(attr)
					plt.title(title, y=1.03, fontsize=15, weight='bold')
					sns.countplot(data[attr])
					st.pyplot(fig)
			else:
				st.warning("The categorical attributes have more than 15 unique values")

		elif data[attr].dtypes != 'object':
			with st.expander("Histogram"):
				fig = plt.figure()
				title = "Distribution of "+str(attr)
				plt.title(title, y=1.03, fontsize=15, weight='bold')
				sns.distplot(data[attr])
				st.pyplot(fig)

			with st.expander("Boxplot"):
				fig = plt.figure()
				sns.boxplot(data[attr])
				st.pyplot(fig)

	elif analysis_type == 'Bivariate':
		col = st.columns(3)
		x = col[0].selectbox('Choose X-axis', data.columns)
		y = col[1].selectbox('Choose Y-axis', data.columns)
		lis = [x for x in data.columns]
		lis.append(None)
		hue = col[2].selectbox('Choose Hue', lis,index = lis.index(None))
		viz_types = st.radio('Choose a chart',('Boxplot','Scatterplot','Lineplot'))
		if x != y:
			if st.button("Plot"):
				if viz_types == 'Scatterplot':
					fig = plt.figure()
					title = str(y)+" vs "+str(x)
					plt.title(title, y=1.03, fontsize=15, weight='bold')
					sns.scatterplot(data=data,x=x,y=y,hue=hue)
					st.pyplot(fig)

				if viz_types == 'Boxplot':
					fig = plt.figure()
					title = str(y)+" vs "+str(x)
					plt.title(title, y=1.03, fontsize=15, weight='bold')
					sns.boxplot(data=data,x=x,y=y,hue=hue)
					st.pyplot(fig)

				if viz_types == 'Lineplot':
					fig = plt.figure()
					title = str(y)+" vs "+str(x)
					plt.title(title, y=1.03, fontsize=15, weight='bold')
					sns.lineplot(data=data,x=x,y=y,hue=hue)
					st.pyplot(fig)

		else:
			st.error("Avoid using same x and y axis attributes")

	else:
		viz_type = st.radio("Please choose",('Correlation Barplot','Pairplot','Heatmap'))
		numeric_data = data.select_dtypes(np.number)
		lis = [x for x in numeric_data.columns]
		lis.append(None)
		if viz_type == 'Correlation Barplot':
			res_var = st.selectbox("Choose response attribute", lis, index = lis.index(None))
			var = st.multiselect('Choose any attribute', lis)
			if (res_var is not None) and (len(var) > 1):
				if st.button("Plot"):
					for_histogram = data.drop(columns=res_var)
					fig = plt.figure()
					for_histogram[var].corrwith(numeric_data[res_var]).plot.bar(title ='Correlation with Response Variable',fontsize=15, rot=45, grid=True)
					st.pyplot(fig)
			else:
				st.warning("Choose one response attribute and more than 2 for the other attribute. ")

		elif viz_type == 'Pairplot':
			col = st.columns(2)
			var = col[0].multiselect('Choose any attribute', lis)
			hue = col[1].selectbox("Choose Hue",lis,index=lis.index(None))
			if len(var) > 1:
				if st.button("Plot"):
					fig = plt.figure()
					st.write(sns.pairplot(data=numeric_data, vars=var, hue=hue, diag_kind='kde'))
					st.pyplot()
			else:
				st.warning("Please choose more than 1 attribute")
		else:
			var = st.multiselect('Choose any attribute', lis)
			if len(var) > 1:
				if st.button("Plot"):
					fig = plt.figure()
					plt.title("Correlation Matrix", y=1.05, fontsize=15, weight='bold')
					st.write(sns.heatmap(numeric_data[var].corr(),annot=True,fmt='.02f',square=True))
					st.pyplot()
			else:
				st.warning("Please choose more than 1 attribute")


# loading function
@st.cache(suppress_st_warning=True)
def load_file(file):
    if file is not None:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            print(e)
            df = pd.read_excel(file)
        return(df)

# the main application
def main():
    selection = ['EDA','Machine learning','About Us']
    option = st.sidebar.selectbox('Main options:',selection)

    if option == "EDA":
        uploaded_file = st.file_uploader("Choose a file (Preferred cleaned dataset)", type = ['csv', 'xlsx'])
        data = load_file(uploaded_file)
        if data is not None:
            st.success("Successfully loaded")
        else:
            st.warning("Please load the dataset")

        st.subheader("Exploratory Data Analysis")
        sub_option = ['About Data','Visualisation']
        sub_option = st.sidebar.selectbox('Sub options:',sub_option)

        if sub_option == 'About Data':
            st.markdown("""
            #####
            ###### Displayed the basic information about the dataset

            """)
            if st.button('Generate report'):
                if data is not None:
                    about_data(data)
                else:
                    st.error('There is no dataset')

        elif sub_option == 'Visualisation':
            st.markdown("""
            #####
            ###### Visualisation features

            """)
            if data is not None:
                viz(data)
            else:
                st.error('There is no dataset')

    elif option == "Machine learning":
        st.subheader("Coming soon")

    else:
        st.markdown("""
        ## About Datama

        ##### Datama is web application used to perform EDA, and visualization for any given dataset. This web application eases the stakeholder's access to any crucial task of every data analytic project.
        ###### Developed by Aiman Raziq


        """)


if __name__ == '__main__':
	main()
