
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np

NOPIX = '0px 0px 0px 0px'
colors = {
    'negative' : '#ef8a62',
    'neutral' : '#f7f7f7',
    'positive' : '#67a9cf',
}

external_sheets = ['https://codepen.io/chriddyp/pen/dZVMbK.css']

app = dash.Dash(__name__)

#______________MAP FIGURE__________________
map_figure = go.Figure(go.Scattermapbox())

map_figure.update_layout(
    hovermode='closest',
    mapbox=go.layout.Mapbox(
        style='open-street-map',
        bearing=0,
        center=go.layout.mapbox.Center(
            lat=37,
            lon=-95,
        ),
        pitch=0,
        zoom=2,
    ),
    margin = {'l': 0, 'b': 0, 't': 0, 'r': 0}
)

#_____________Tweet Activity Meters__________

activity_meter = make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.05)

activity_meter.append_trace(go.Scatter(marker_color = colors['positive']), row=1,col=1)
activity_meter.append_trace(go.Scatter(marker_color = colors['positive']), row=2,col=1)

activity_meter.update_layout(
    margin = {'l': 0, 'b': 0, 't': 25, 'r': 0},
    font = {'size' : 9, 'family' : "Arial"},
    plot_bgcolor = '#F0F0F0',
)
activity_meter.update_xaxes(title_text='Time (mins)', row=2,col=1)
activity_meter.update_yaxes(title_text='Total Tweets Processed',row=1,col=1)
activity_meter.update_yaxes(title_text='Tweets/min',row=2,col=1)

latency = go.Figure(go.Scatter(marker_color = colors['negative']))
latency.update_layout(
    title = 'Stream Latency',
    margin = {'l': 0, 'b': 0, 't': 25, 'r': 0},
    font = dict(size = 9, family = 'Arial'),
    plot_bgcolor = '#F0F0F0',
    xaxis_title = 'Tweets',
    yaxis_title = 'Latency (seconds)',
)

#__________Sentiment Meters_________________

sentiment_river = go.Figure()
sentiment_river.add_trace(go.Scatter(
    mode = 'lines',
    marker_color = colors['negative'],
    stackgroup='one',
    groupnorm='percent'
))

sentiment_river.add_trace(go.Scatter(
    mode = 'lines',
    fillcolor = colors['neutral'],
    line = dict(width=0.5, color=colors['positive']),
    stackgroup='one',
))

sentiment_river.add_trace(go.Scatter(
    mode = 'lines',
    line = dict(width=0.5, color=colors['positive']),
    stackgroup='one',
))

sentiment_river.update_layout(
    showlegend = False,
    #xaxis_type = 'category'
    plot_bgcolor = '#F0F0F0',
    margin = {'l': 0, 'b': 0, 't': 25, 'r': 0},
    xaxis_title = 'Time (mins)',
    yaxis=dict(
        type='linear',
        range=[1, 100],
        ticksuffix='%'),
    font = dict(size = 9, family = 'Arial'),
)

x= ['']
cumm_sentiment = go.Figure()
cumm_sentiment.add_trace(go.Bar(x= x, y = [4], marker_color = colors['negative'], name = 'Negative'))
cumm_sentiment.add_trace(go.Bar(x= x, y = [4], marker_color = colors['neutral'], name = 'Neutral'))
cumm_sentiment.add_trace(go.Bar(x= x, y = [4], marker_color = colors['positive'], name = 'Positive'))

cumm_sentiment.update_layout(
    title = 'Cummulative Sentiment',
    yaxis_title = 'Num Tweets',
    showlegend = True,
    #xaxis_type = 'category'
    plot_bgcolor = '#F0F0F0',
    margin = {'l': 0, 'b': 0, 't': 25, 'r': 25},
    barmode = 'stack',
    font = dict(size = 9, family = 'Arial'),
    xaxis_title = '',
)

#_________Tweets Table__________________________

tweet_sample_columns = [
    'Sentiment', 'Text'
]
tweets_table = go.Figure([go.Table(
    columnwidth = [40,400],
    header=dict(values=list(tweet_sample_columns),
                fill_color='lightgrey',
                align='left',
                ),
    cells=dict(values=[['+', '-','0','-'], [['Hey this is a tweet'], ['Whoooooo I really love to send fake tweets'],
                                ['What\'s the character limit again? 140?'],['Im gonna be the president']]],
               fill_color= 'white',
               align='left'))
])
tweets_table.update_layout(
    margin = {'l': 0, 'b': 0, 't': 5, 'r': 0},
)


with open('./assets/citynames.txt', 'r') as f:
    citynames = f.readlines()

#'0px 10px 25px 25px'
app.layout = html.Div([

    html.Div([
        html.Div(
            html.Img(
                src=app.get_asset_url('twitter_icon.png'),
                height=70,
                width=70,
            ),
            style = {'display' : 'inline-block'},
        ),
        html.Div([
           html.H1('Vibe', style = {'margin-bottom' : 0, 'padding-bottom' : 0}),
           html.H6('How\'s Twitter Feelin?', style = {'margin-top':0,'padding-top':0})
        ],
        style = {'display' : 'inline-block', 'margin-left' : 25, 'margin-top' : 15, 'align':'center'}),
    ]),

    html.Hr(style = {'margin' : '0px 25px 25px 0px', 'padding' : '0px 25px 0px 0px'}),
    
    html.Div([
#______________LEFT PANEL _____________________
        html.Div([
            html.Label('Monitor City'),
            dcc.Dropdown(
                    options=[
                        {'label' : name, 'value' : name} for name in citynames
                    ],
                    value='Minneapolis',
            ),
            html.Br(),
            html.Div([
                html.Div([
                    #html.Label('Location Tags'),
                    html.H6('Geo Tags', className = 'infobox_header'),
                    dcc.Graph(
                        id = 'tweet_map',
                        figure = map_figure,
                        style = {'width':'100%','height':'40vh'}
                    ),
                ], style = {'width':'46%', 'float':'left'}, className = 'infobox',),
                html.Div([
                    html.H6('Activity', className = 'infobox_header'),
                    dcc.Graph(
                        id = 'activity',
                        figure = activity_meter,
                        style = {'width' : '100%','height' : '40vh'},
                    )
                ], style ={'width':'46%','float':'right'}, className = 'infobox'),
                html.Div(style = {'clear' : 'both'}),
                html.Div([
                    html.H6('Stream Status', className = 'infobox_header'),
                    html.Div([
                        dcc.Graph(
                            id = 'stream_latency',
                            figure = latency,
                            style = {'height' : '20vh', 'margin-top' : 0, 'padding-top' : 0}
                        ),
                    ], style = dict(width = '48%', float = 'right')),
                    html.Div([
                        html.Label('  Attenuation',style = {'font-size' : 14}),
                        html.Br(),
                        dcc.Slider(
                            min = 0,
                            max = 15,
                            marks = {
                                i : str(i) for i in range(0,15+1)
                            },
                            value = 0,
                        ),
                        dcc.Markdown('''
                                Attenuating the stream by a factor of **a = 2^x**, meaning the app samples tweets from the stream with uniform probability 
                                **p = 1/a**. This may help reduce strain on the input pipeline if volume is too high.
                        ''')
                    ], style = {'float' : 'left','width' : '48%'}),
                    html.Div(style = {'clear' : 'both'})
                ], className = 'infobox', style = {'clear' : 'both', 'margin-top' : 25}),
            ]),            
        ], style = {'float' : 'left', 'width' : '45%', 'margin' : '0px 0px 25px 25px'}),

#______________RIGHT PANEL _____________________
        html.Div([
            html.Div([
                html.H6('Sentiment', className = 'infobox_header'),
                html.Div([
                    html.Div([
                        dcc.Graph(
                            id = 'sentiment_river',
                            figure = sentiment_river,
                            style = {
                                'height' : '30vh', 'padding-right' : 15,
                            }
                        ),
                    ], style = {'width' : '75%', 'float':'left',}),
                    html.Div([
                        dcc.Graph(
                            id = 'cummulative',
                            figure = cumm_sentiment,
                            style = {'height' : '30vh'}
                        ),
                        html.Label('Geo-tags'),
                        dcc.Dropdown(
                            options = [{'label' : 'All', 'value' : 'All'}],
                            value = ['All'],
                            multi = True,
                        ),
                        html.Label('Subjects'),
                        dcc.Dropdown(
                            options = [{'label' : 'All', 'value' : 'All'}],
                            value = ['All'],
                            multi = True,
                        ),
                        html.Label('Keyword'),
                        dcc.Input(value='',type = 'text')              
                    ], style = {'width' : '25%', 'float': 'right'}),]),
                html.Div(style={'clear':'both'})
            ], className = 'infobox', style = {'margin-bottom' : 25}),
            html.Div([
                html.H6('Classification Samples', className = 'infobox_header', style = {'margin-bottom' : 5, 'display' : 'inline'}),
                html.Button('Shuffle', id='refresh', style = {'float' : 'right', 'display' : 'inline'}),
                html.Div(style = {'clear' : 'right'}),
                dcc.Graph(
                    id= 'tweets_table',
                    figure = tweets_table,
                    style = {'padding' : 0,}
                )
            ], className = 'infobox')
        ], style = {'float' : 'right', 'width' : '51%', 'margin' : 23})
    ])
])

if __name__ == "__main__":
    app.run_server(debug=True)