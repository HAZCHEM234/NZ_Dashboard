import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import statsmodels.api as sm
import os

# Load data
merged_data = pd.read_csv('https://raw.githubusercontent.com/HAZCHEM234/NZ_data/main/data/merged_data.csv')
immigration_data = pd.read_csv('https://raw.githubusercontent.com/HAZCHEM234/NZ_data/main/data/2018-2024%20Immigration%20.csv')
house_price_index = pd.read_csv('https://raw.githubusercontent.com/HAZCHEM234/NZ_data/main/data/House_price_csv_mode.csv')
labor_market = pd.read_csv('https://raw.githubusercontent.com/HAZCHEM234/NZ_data/main/data/Labor%20market%20clean.csv')

# Convert date columns to datetime
merged_data['Year'] = pd.to_datetime(merged_data['Year'], format='%Y')
immigration_data['Date'] = pd.to_datetime(immigration_data['Date'], format='%d/%m/%Y')
house_price_index['Date'] = pd.to_datetime(house_price_index['Date'], format='%d/%m/%Y')
labor_market['Date'] = pd.to_datetime(labor_market['Date'], format='%d/%m/%Y')

# Create initial figures
def create_figures():
    fig1 = px.line(merged_data, x='Year', y='Unemployment rate', title='Sum of Unemployment Rate by Year', template='plotly_dark')
    fig2 = px.line(merged_data, x='Year', y='House price index (HPI)', title='Sum of House Price Index (HPI) by Year', template='plotly_dark')
    fig3 = px.line(merged_data, x='Year', y='estimate', title='Sum of Estimates Immigration by Year', template='plotly_dark')
    fig4 = px.scatter(merged_data, x='Unemployment rate', y='House price index (HPI)', trendline='ols', title='Sum of Unemployment vs HPI', template='plotly_dark')
    fig5 = px.scatter(merged_data, x='estimate', y='House price index (HPI)', trendline='ols', title='Sum of Estimate Immigration vs HPI', template='plotly_dark')
    
    immigration_filtered = immigration_data[
        (immigration_data['visa'] != 'TOTAL') & 
        (immigration_data['citizenship'] != 'TOTAL') & 
        (immigration_data['country_of_residence'] != 'TOTAL')
    ]
    
    fig6 = px.pie(immigration_filtered, names='visa', values='estimate', title='Estimate by Visa (Excluding TOTAL)', template='plotly_dark')
    fig7 = px.pie(immigration_filtered, names='citizenship', values='estimate', title='Estimate by Citizenship (Excluding TOTAL)', template='plotly_dark')
    
    country_residence_sorted = (
        immigration_filtered
        .groupby('country_of_residence')['estimate']
        .sum()
        .nlargest(10)
        .reset_index()
    )
    
    fig8 = px.bar(
        country_residence_sorted,
        y='country_of_residence',
        x='estimate',
        orientation='h',
        title='Largest Estimate by Country of Residence (Excluding TOTAL)',
        text='estimate',
        template='plotly_dark'
    )
    fig8.update_layout(yaxis={'categoryorder':'total ascending'})
    
    fig9 = px.scatter(house_price_index, x='House price index (HPI)', y='Total value of housing stock ', trendline='ols', title='HPI vs Total Value of Housing Stocks', template='plotly_dark')
    fig10 = px.line(house_price_index, x='Date', y='House Sale', title='House Sales Over Time', template='plotly_dark')
    fig11 = px.line(house_price_index, x='Date', y='Residential investment (GDP)', title='Residential Investment Over Time', template='plotly_dark')
    
    labor_market_filtered = labor_market[labor_market['Date'] >= '2018-01-01']
    
    fig12 = px.line(labor_market_filtered, x='Date', y='Unemployment rate %', title='Unemployment Rate vs Date (2018 to Current)', template='plotly_dark')
    fig13 = px.line(labor_market_filtered, x='Date', y='Average hourly earnings (ordinary time and overtime)', title='Average Hourly Rate vs Date (2018 to Current)', template='plotly_dark')
    fig14 = px.line(labor_market_filtered, x='Date', y='Working-age population %', title='Working Age Population vs Date (2018 to Current)', template='plotly_dark')
    fig15 = px.scatter(labor_market_filtered, x='Unemployment rate %', y='Working-age population %', trendline='ols', title='Unemployment Rate vs Working Age Population (2018 to Current)', template='plotly_dark')
    fig16 = px.scatter(labor_market_filtered, x='Unemployment rate %', y='Average hourly earnings (ordinary time and overtime)', trendline='ols', title='Unemployment Rate vs Average Hourly Earnings (2018 to Current)', template='plotly_dark')
    fig17 = px.line(labor_market_filtered, x='Date', y='Total value of housing stock', title='Housing Stock vs Date', template='plotly_dark')
    
    return fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11, fig12, fig13, fig14, fig15, fig16, fig17

app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div(style={'backgroundColor': '#1f2c56', 'fontFamily': 'Arial, sans-serif'}, children=[
    html.H1("Economic and Housing Data Dashboard", style={'textAlign': 'center', 'color': '#ffffff'}),
    
    dcc.Tabs([
        dcc.Tab(label='Sum of Yearly Unemployment, Immigration & Housing Data', children=[
            html.Div([
                dcc.Graph(figure=create_figures()[0], style={'width': '48%', 'display': 'inline-block'}),
                dcc.Graph(figure=create_figures()[1], style={'width': '48%', 'display': 'inline-block'}),
                dcc.Graph(figure=create_figures()[2], style={'width': '48%', 'display': 'inline-block'}),
                dcc.Graph(figure=create_figures()[3], style={'width': '48%', 'display': 'inline-block'}),
                dcc.Graph(figure=create_figures()[4], style={'width': '48%', 'display': 'inline-block'})
            ])
        ]),
        dcc.Tab(label='Immigration Data', children=[
            html.Div([
                dcc.Graph(figure=create_figures()[5], style={'width': '48%', 'display': 'inline-block'}),
                dcc.Graph(figure=create_figures()[6], style={'width': '48%', 'display': 'inline-block'}),
                dcc.Graph(figure=create_figures()[7], style={'width': '48%', 'display': 'inline-block'})
            ])
        ]),
        dcc.Tab(label='Housing & Labor Market Data', children=[
            html.Div([
                dcc.Graph(figure=create_figures()[8], style={'width': '48%', 'display': 'inline-block'}),
                dcc.Graph(figure=create_figures()[13], style={'width': '48%', 'display': 'inline-block'}),
                dcc.Graph(figure=create_figures()[16], style={'width': '48%', 'display': 'inline-block'}),
                dcc.Graph(figure=create_figures()[12], style={'width': '48%', 'display': 'inline-block'}),
                dcc.Graph(figure=create_figures()[11], style={'width': '48%', 'display': 'inline-block'}),
                dcc.Graph(figure=create_figures()[9], style={'width': '48%', 'display': 'inline-block'}),
                dcc.Graph(figure=create_figures()[10], style={'width': '48%', 'display': 'inline-block'}),
            ])
        ])
    ])
])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
