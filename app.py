from dash import Dash, dcc, html, Input, Output
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
import dash_bootstrap_components as dbc
import os


# Define the custom template
custom_template = go.layout.Template(
    layout=dict(
        font=dict(family="Arial", size=12, color="black"),
        title=dict(font=dict(family="Arial", size=20, color="black")),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='black',
            ticks='outside'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            zeroline=False,
            showline=True,
            linewidth=2,
            linecolor='black',
            ticks='outside'
        ),
        legend=dict(
            font=dict(family="Arial", size=12, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
    )
)

# Define a dictionary to map ticker symbols to company names
ticker_to_name = {
    'AIR.NZ': 'Air New Zealand',
    'UAL': 'United Airlines',
    'QAN.AX': 'Qantas',
    'DAL': 'Delta'
}

# Function to plot stock prices
def plot_stock_vs_market(ticker_stock, start_date='2020-01-01', end_date='2024-01-01'):
    data_stock = yf.download(ticker_stock, start=start_date, end=end_date)
    company_name = ticker_to_name.get(ticker_stock, ticker_stock)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data_stock.index,
        y=data_stock['Adj Close'],
        mode='lines',
        name=ticker_stock,
        line=dict(color='#7AE582')
    ))

    fig.update_layout(
        title=f'{company_name} (2020-2024)',
        xaxis_title='Date',
        yaxis_title='Adjusted Close Price',
        legend_title='Ticker',
        template=custom_template
    )

    return fig

def plot_differenced_closing_prices(ticker, start_date='2020-01-01', end_date='2024-01-01'):
    # Fetch historical market data for the given ticker
    data_stock = yf.download(ticker, start=start_date, end=end_date)
    data = data_stock['Close']
    data_diff = data.diff().dropna()

    # Get the company name based on the ticker
    company_name = ticker_to_name.get(ticker, ticker)  # Default to ticker if not found

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data_diff.index,
        y=data_diff,
        mode='lines',
        name=f'{ticker} Differenced Close Price',
        line=dict(color='#25A18E')
    ))

    # Update layout
    fig.update_layout(
        title=f'Differenced {company_name} Closing Prices (2020-2024)',
        xaxis_title='Date',
        yaxis_title='Differenced Close Price',
        legend_title='Price Type',
        template=custom_template
    )

    return fig


def plot_acf_plotly(ticker, start_date='2020-01-01', end_date='2024-01-01', lags=50):
    # Fetch historical market data for the given ticker
    data_stock = yf.download(ticker, start=start_date, end=end_date)
    
    # Use the 'Close' price
    data = data_stock['Close']
    
    # Differencing to make the data stationary
    data_diff = data.diff().dropna()
    
    # Calculate ACF
    acf_values = acf(data_diff, nlags=lags)
    
    # Get the company name based on the ticker
    company_name = ticker_to_name.get(ticker, ticker)  # Default to ticker if not found
    
    # Create ACF plot
    fig = go.Figure(go.Bar(
        x=list(range(len(acf_values))),
        y=acf_values,
        name='ACF',
        marker_color='#9FFFCB'
    ))

    # Update layout
    fig.update_layout(
        title=f'ACF of Differenced {company_name} Data (2020-2024)',
        xaxis_title='Lags',
        yaxis_title='Correlation',
        template=custom_template
    )

    return fig

def plot_pacf_plotly(ticker, start_date='2020-01-01', end_date='2024-01-01', lags=50):
    # Fetch historical market data for the given ticker
    data_stock = yf.download(ticker, start=start_date, end=end_date)
    data = data_stock['Close']
    data_diff = data.diff().dropna()
    
    # Calculate PACF
    pacf_values = pacf(data_diff, nlags=lags)
    
    # Get the company name based on the ticker
    company_name = ticker_to_name.get(ticker, ticker)  # Default to ticker if not found
    
    # Create PACF plot
    fig = go.Figure(go.Bar(
        x=list(range(len(pacf_values))),
        y=pacf_values,
        name='PACF',
        marker_color='#004E64'
    ))

    # Update layout
    fig.update_layout(
        title=f'PACF of Differenced {company_name} Data (2020-2024)',
        xaxis_title='Lags',
        yaxis_title='Correlation',
        template=custom_template
    )

    return fig

def plot_arima_forecast(ticker, start_date='2020-01-01', end_date='2024-01-01', p=1, d=1, q=1, forecast_steps=10):
    data_stock = yf.download(ticker, start=start_date, end=end_date)
    data = data_stock['Close']

    # Set frequency to daily ('D') and forward fill missing data
    data = data.asfreq('D').ffill()

    model = ARIMA(data, order=(p, d, q))
    model_fit = model.fit()

    # Forecast future values
    forecast = model_fit.forecast(steps=forecast_steps)
    forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D')

    # Create figure
    fig = go.Figure()

    # Add historical data trace
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data,
        mode='lines',
        name='Historical Close Price',
        line=dict(color='#00A5CF')
    ))

    # Add forecasted data trace
    fig.add_trace(go.Scatter(
        x=forecast_index,
        y=forecast,
        mode='lines',
        name='Forecasted Close Price',
        line=dict(color='red')
    ))

    # Update layout
    company_name = ticker_to_name.get(ticker, ticker)  # Default to ticker if not found
    fig.update_layout(
        title=f'{company_name} Close Price Forecast',
        xaxis_title='Date',
        yaxis_title='Close Price',
        legend_title='Data',
        template=custom_template
    )

    return fig


# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.H1(
        "Airline Stock Price Analysis Using ARIMA: ACF and PACF Insights",
        style={
            'textAlign': 'center',  # Center aligns the title
            'marginBottom': '4vw',
            'marginTop': '4vw'
        }
    ),
    dcc.Dropdown(
        id='ticker-dropdown',
        options=[{'label': name, 'value': ticker} for ticker, name in ticker_to_name.items()],
        value='AIR.NZ',  # Default value
        style={'width': '50%', 'marginBottom': '4vw', 'marginLeft': '1vw'}  # Adds space below the dropdown
    ),
    dbc.Row([
        dbc.Col(dcc.Graph(id='stock-graph'), width=6),
        dbc.Col(dcc.Graph(id='arima-graph'), width=6),
    ], style={'marginBottom': '30px'}),  # Adds space below the first row of graphs
    dbc.Row([
        dbc.Col(dcc.Graph(id='acf-graph'), width=6),
        dbc.Col(dcc.Graph(id='pacf-graph'), width=6),
    ], style={'marginBottom': '30px'}),  # Adds space below the second row of graphs
    dbc.Row([
        dbc.Col(dcc.Graph(id='differenced-graph'), width=12),
    ])
])


@app.callback(
    [Output('stock-graph', 'figure'),
     Output('differenced-graph', 'figure'),
     Output('acf-graph', 'figure'),
     Output('pacf-graph', 'figure'),
     Output('arima-graph', 'figure')],
    [Input('ticker-dropdown', 'value')]
)
def update_graph(selected_ticker):
    stock_fig = plot_stock_vs_market(selected_ticker)
    differenced_fig = plot_differenced_closing_prices(selected_ticker)
    acf_fig = plot_acf_plotly(selected_ticker)
    pacf_fig = plot_pacf_plotly(selected_ticker)
    arima_fig = plot_arima_forecast(selected_ticker)
    
    return stock_fig, differenced_fig, acf_fig, pacf_fig, arima_fig


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
