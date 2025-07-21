from dash import Dash, html, dcc, Input, Output
import plotly.graph_objects as go
import numpy as np
from fun import *


options_LIF = {
    'tau_V': 20,
    'R': 20,
    'V_L': 0,
    'I_stim_on': 1000,
    'I_stim_off': 2000,
    'I_stim': 0.9,
    'V_th': 20,
    'V_r': 0,
    'tau_ref': 5,
    'dt': 0.05,
    't_total': 5000
}


options_DDM = {
    'nu': 0.4,
    's': 0.2,
    'sigma': 4,
    'y_0': 0,
    'lower_bound': -0.25,
    'upper_bound': 0.25,
    'dt': 0.05,
    't_total': 5000
}


# Network plots
binfired = sim_network()

x = np.arange(0, 2, 0.05)
y = np.zeros(len(x))
for i, x_i in enumerate(x):
    r = ((binfired[:, 0] / 1000 > x_i) & (binfired[:, 0] / 1000 <= x_i + 0.05)).sum() / 50 / 0.05
    y[i] = r
fig1 = go.Figure(data=go.Scatter(x=x + 0.025, y=y, mode='lines'))
fig1.update_layout(
    height=200,
    xaxis={'range': [0, 2], 'autorange': False},
    margin=dict(l=100, r=0, b=10, t=0), 
    yaxis_title='Firing rate'
)

fig2 = go.Figure(data=go.Scatter(
    x=binfired[:, 0] / 1000, 
    y=binfired[:, 1], 
    mode='markers', 
    marker=dict(size=3, color='blue')))
fig2.update_layout(
    height=200,
    xaxis={'range': [0, 2], 'autorange': False},
    yaxis={'range': [0, 49], 'autorange': False},
    margin=dict(l=100, r=0, b=10, t=0), 
    yaxis_title='Spikes'
)

x = np.arange(0, 2, 0.001)
y = np.zeros(len(x))
for i, x_i in enumerate(x):
    if (x_i >= 0.3) and (x_i <= 0.5):
        y[i] = 1
    elif (x_i >= 1.4) and (x_i <= 1.6):
        y[i] = -1
fig3 = go.Figure(data=go.Scatter(x=x, y=y, mode='lines'))
fig3.update_layout(
    height=200,
    xaxis={'range': [0, 2], 'autorange': False},
    margin=dict(l=100, r=0, b=10, t=0), 
    xaxis_title=r'$t$',
    yaxis_title=r'$I^\mathrm{(ext)}$'
)

# main layout
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = 'Simulations'

app.layout = html.Div([

    html.H1('Computational neuroscience model demos'),
    html.Hr(),
    
    # LIF NEURON
    html.H3('Leaky Integrate-and-Fire neuron (stimulus-driven)'),

    html.Div([
        dcc.Markdown(children="""
            $$
            \\tau_\\mathrm{V} \\frac{\\mathrm{d}V}{\\mathrm{d}t} = -(V - V_\\mathrm{L}) + IR         
            $$
            $$
            V \\ge V_\\mathrm{th} \\rightarrow V = V_\\mathrm{r} \\ \\ \\mathrm{for} \\ \\ \\tau_\\mathrm{ref}         
            $$
            """, mathjax=True)
    ]),

    html.Div([

        # Options panel
        html.Div([

            html.Div([dcc.Markdown(children='$I$: Stimulus time', mathjax=True)]),
            dcc.RangeSlider(0, options_LIF['t_total'] / 1000,
                            value=[options_LIF['I_stim_on'] / 1000, options_LIF['I_stim_off'] / 1000],
                            marks={
                                0: {'label': '0 s'},
                                int(options_LIF['I_stim_on'] / 1000): {'label': '{:.0f} s'.format(options_LIF['I_stim_on'] / 1000)},
                                int(options_LIF['I_stim_off'] / 1000): {'label': '{:.0f} s'.format(options_LIF['I_stim_off'] / 1000)},
                                int(options_LIF['t_total'] / 1000): {'label': '{:.0f} s'.format(options_LIF['t_total'] / 1000)}
                            }, id='widget_stimulus_time'),

            html.Div([dcc.Markdown(children='$I$: Stimulus strength', mathjax=True)]),
            dcc.Slider(0, 10, value=options_LIF['I_stim'], marks={
                0: {'label': '0 pA'},
                options_LIF['I_stim']:{'label':'{} pA'.format(options_LIF['I_stim'])},
                10: {'label': '10 pA'}
            }, id='widget_stimulus_strength'),

            html.Div([dcc.Markdown(children='$\\tau_\\mathrm{V}$', mathjax=True)]),
            dcc.Slider(1, 100, value=options_LIF['tau_V'], marks={
                1: {'label': '1 ms'},
                options_LIF['tau_V']: {'label': '{} ms'.format(options_LIF['tau_V'])},
                100: {'label': '100 ms'}
            }, id='widget_tau_V'),

            html.Div([dcc.Markdown(children='$R$', mathjax=True)]),
            dcc.Slider(0, 100, value=options_LIF['tau_V'], marks={
                0: {'label': '0 MOhm'},
                options_LIF['R']: {'label': '{} MOhm'.format(options_LIF['R'])},
                100: {'label': '100 MOhm'}
            }, id='widget_R'),

            html.Div([dcc.Markdown(children='$V_\\mathrm{L}$', mathjax=True)]),
            dcc.Slider(-50, 50, value=options_LIF['V_L'], marks={
                -50: {'label': '-50 mV'},
                0: {'label': '0 mV'},
                50: {'label': '50 mV'}
            }, id='widget_V_L'),

            html.Div([dcc.Markdown(children='$V_\\mathrm{th}$', mathjax=True)]),
            dcc.Slider(0, 80, value=options_LIF['V_th'], marks={
                0: {'label': '0 mV'},
                options_LIF['V_th']: {'label': '{} mV'.format(options_LIF['V_th'])},
                80: {'label': '80 mV'}
            }, id='widget_V_th'),

            html.Div([dcc.Markdown(children='$V_\\mathrm{r}$', mathjax=True)]),
            dcc.Slider(-50, 50, value=options_LIF['V_r'], marks={
                -50: {'label': '-50 mV'},
                0: {'label': '0 mV'},
                50: {'label': '50 mV'}
            }, id='widget_V_r'),

            html.Div([dcc.Markdown(children='$\\tau_\\mathrm{ref}$', mathjax=True)]),
            dcc.Slider(0, 50, value=options_LIF['tau_ref'], marks={
                0: {'label': '0 ms'},
                options_LIF['tau_ref']: {'label': '{} ms'.format(options_LIF['tau_ref'])},
                50: {'label': '50 ms'}
            }, id='widget_tau_ref'),
        
        ], style={'height': '500px', 'width': '500px'}),

        # Figures
        html.Div([

            html.Div([
                html.H3(''),
                dcc.Graph(id='g1_spikes', mathjax=True)
            ]),

            html.Div([
                html.H3(''),
                dcc.Graph(id='g1_V', mathjax=True)
            ]),

            html.Div([
                html.H3(''),
                dcc.Graph(id='g1_I', mathjax=True)
            ])
        
        ], style={'height': '1000px', 'width': '1000px'})

    ], style={'display': 'flex'}),

    # LIF NETWORK
    html.H3('Leaky Integrate-and-Fire network'),

    html.Div([
        dcc.Markdown(children="""
            $$
            \\tau_\\mathrm{I} \\frac{\\mathrm{d}I_i^\\mathrm{(syn)}}{\\mathrm{d}t} = -I_i^\\mathrm{(syn)} + \\sum_{j,n} J_{ij} \\delta(t - s_{j,n})         
            $$
            $$
            \\tau_\\mathrm{V} \\frac{\\mathrm{d}V_i}{\\mathrm{d}t} = -(V_i - V_\\mathrm{L}) + (I^\\mathrm{(ext)} + I_i^\\mathrm{(syn)})R         
            $$
            $$
            V_i \\ge V_\\mathrm{th} \\rightarrow V_i = V_\\mathrm{r} \\ \\ \\mathrm{for} \\ \\ \\tau_\\mathrm{ref}         
            $$
            """, mathjax=True)
    ]),

    html.Div([

        # Figures
        html.Div([

            html.Div([
                html.H3(''),
                dcc.Graph(id='g2_rates', mathjax=True, figure=fig1)
            ]),

            html.Div([
                html.H3(''),
                dcc.Graph(id='g2_spikes', mathjax=True, figure=fig2)
            ]),

            html.Div([
                html.H3(''),
                dcc.Graph(id='g2_inputs', mathjax=True, figure=fig3)
            ])
        
        ], style={'height': '1000px', 'width': '1000px'})

    ], style={'display': 'flex'}),

    # DRIFT-DIFFUSION MODEL
    html.H3('Drift-diffusion model'),

    html.Div([
        dcc.Markdown(children="""
            $$
            \\mathrm{d}y = \\nu \\ \\mathrm{d}t + s \\ N(0, \\sigma) \\ \\sqrt{\\mathrm{d}t}         
            $$
            """, mathjax=True)
    ]),

    html.Div([

        # Options panel
        html.Div([

            html.Div([dcc.Markdown(children='$\\nu$', mathjax=True)]),
            dcc.Slider(-1, 1, value=options_DDM['nu'], marks={
                -1: {'label': '-1'},
                0: {'label': '0'},
                options_DDM['nu']: {'label': '{}'.format(options_DDM['nu'])},
                1: {'label': '1'}
            }, id='widget_nu'),

            html.Div([dcc.Markdown(children='$s$', mathjax=True)]),
            dcc.Slider(0, 5, value=options_DDM['s'], marks={
                0: {'label': '0'},
                options_DDM['s']: {'label': '{}'.format(options_DDM['s'])},
                5: {'label': '5'}
            }, id='widget_s'),

            html.Div([dcc.Markdown(children='$\\sigma$', mathjax=True)]),
            dcc.Slider(0, 10, value=options_DDM['sigma'], marks={
                0: {'label': '0'},
                options_DDM['sigma']: {'label': '{}'.format(options_DDM['sigma'])},
                10: {'label': '10'}
            }, id='widget_sigma'),

            html.Div([dcc.Markdown(children='$y_0$', mathjax=True)]),
            dcc.Slider(-0.24, 0.24, value=options_DDM['y_0'], marks={
                -0.24: {'label': '-0.24'},
                0: {'label': '0'},
                0.24: {'label': '0.24'}
            }, id='widget_y0'),

            html.Div([
                html.Button('Re-run', id='re_run_button', style={'width': '100px', 'height': '30px'})
            ], style={'padding': '10px', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'})
        
        ], style={'height': '500px', 'width': '500px'}),

        # Figures
        html.Div([

            html.Div([
                html.H3(''),
                dcc.Graph(id='g2', mathjax=True)
            ])
        
        ], style={'height': '1000px', 'width': '1000px'})

    ], style={'display': 'flex'})

])


# LIF plot updating behavior
@app.callback(
    [Output('g1_spikes', 'figure'),
     Output('g1_V', 'figure'),
     Output('g1_I', 'figure')],
    [Input('widget_stimulus_time', 'value'),
     Input('widget_stimulus_strength', 'value'),
     Input('widget_tau_V', 'value'),
     Input('widget_R', 'value'),
     Input('widget_V_L', 'value'),
     Input('widget_V_th', 'value'),
     Input('widget_V_r', 'value'),
     Input('widget_tau_ref', 'value')]
)
def update_LIF_plot(
    stimulus_time,
    stimulus_strength,
    tau_V,
    R,
    V_L,
    V_th,
    V_r,
    tau_ref):
    # update options
    options_LIF['I_stim_on'], options_LIF['I_stim_off'] = [1000 * t for t in stimulus_time]
    options_LIF['I_stim'] = stimulus_strength
    options_LIF['tau_V'] = tau_V
    options_LIF['R'] = R
    options_LIF['V_L'] = V_L
    options_LIF['V_th'] = V_th
    options_LIF['V_r'] = V_r
    options_LIF['tau_ref'] = tau_ref
    # simulate
    res = sim_LIF(options_LIF)
    # plot
    fig1 = go.Figure(data=go.Scatter(
        x=[t / 1000 for t in res['spike_times']],
        y=[1 for _ in range(len(res['spike_times']))], 
        mode='markers'))
    fig1.update_layout(
        height=200,
        xaxis={'range': [res['t'][0] / 1000, res['t'][-1] / 1000], 'autorange': False},
        yaxis={'showticklabels': False},
        margin=dict(l=100, r=0, b=50, t=0), 
        yaxis_title='Spikes'
    )
    fig2 = go.Figure(data=go.Scatter(x=res['t'] / 1000, y=res['V'], mode='lines'))
    fig2.add_hline(y=options_LIF['V_r'], line_width=1, line_dash='dot', annotation={'text': r'$V_\mathrm{r}$', 'x': 0.7})
    fig2.add_hline(y=options_LIF['V_L'], line_width=1, line_dash='dot', annotation={'text': r'$V_\mathrm{L}$', 'x': 0.8})
    fig2.add_hline(y=options_LIF['V_th'], line_width=1, line_dash='dot', annotation={'text': r'$V_\mathrm{th}$', 'x': 0.9})
    fig2.update_layout(
        height=200,
        xaxis={'range': [res['t'][0] / 1000, res['t'][-1] / 1000], 'autorange': False},
        margin=dict(l=100, r=0, b=50, t=0), 
        yaxis_title=r'$V$'
    )
    fig3 = go.Figure(data=go.Scatter(x=res['t'] / 1000, y=res['I'], mode='lines'))
    fig3.update_layout(
        height=200,
        xaxis={'range': [res['t'][0] / 1000, res['t'][-1] / 1000], 'autorange': False},
        margin=dict(l=100, r=0, b=50, t=0), 
        xaxis_title=r'$t$',
        yaxis_title=r'$I$'
    )
    
    return fig1, fig2, fig3


# DDM plot updating behavior
@app.callback(
    Output('g2', 'figure'),
    [Input('widget_nu', 'value'),
     Input('widget_s', 'value'),
     Input('widget_sigma', 'value'),
     Input('widget_y0', 'value'),
     Input('re_run_button', 'n_clicks')]
)
def update_DDM_plot(
    nu,
    s,
    sigma,
    y_0,
    _):
    # update options
    options_DDM['nu'] = nu
    options_DDM['s'] = s
    options_DDM['sigma'] = sigma
    options_DDM['y_0'] = y_0
    # simulate
    res = sim_DDM(options_DDM)
    # plot
    mask = ~np.isnan(res['y'])
    fig = go.Figure(data=go.Scatter(x=res['t'][mask] / 1000, y=res['y'][mask], mode='lines'))
    fig.add_hline(y=options_DDM['upper_bound'], line_width=1, line_dash='dot')
    fig.add_hline(y=options_DDM['lower_bound'], line_width=1, line_dash='dot')
    if res['decision_point'] is not None:
        x, y = res['decision_point']
        fig.add_trace(go.Scatter(x=[x / 1000], y=[y]))
    fig.update_layout(
        height=200,
        xaxis={'range': [0, options_DDM['t_total'] / 1000], 'autorange': False},
        yaxis={'range': [options_DDM['lower_bound'] - 0.1, options_DDM['upper_bound'] + 0.1], 'autorange': False},
        margin=dict(l=100, r=0, b=50, t=0), 
        xaxis_title=r'$t$',
        yaxis_title=r'$y$',
        showlegend=False
    )
    
    return fig


if __name__ == '__main__':
    app.run(debug=True)