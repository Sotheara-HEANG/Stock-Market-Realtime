-- Dimension: Indicator
-- One row per indicator / source combination with metadata.

CREATE TABLE IF NOT EXISTS gold.dim_indicator (
    indicator_key   SERIAL       PRIMARY KEY,
    indicator_name  VARCHAR(100) NOT NULL,
    source          VARCHAR(100) NOT NULL,
    category        VARCHAR(50),            -- governance | economic | democracy | human_development
    unit            VARCHAR(50),
    description     TEXT,
    UNIQUE (indicator_name, source)
);

INSERT INTO gold.dim_indicator (indicator_name, source, category, unit, description) VALUES
-- WGI governance
('control_of_corruption',          'WGI',     'governance',         'score',       'Perceptions of extent to which public power is exercised for private gain'),
('government_effectiveness',       'WGI',     'governance',         'score',       'Quality of public services and policy formulation'),
('political_stability',            'WGI',     'governance',         'score',       'Likelihood of political instability and violence'),
('regulatory_quality',             'WGI',     'governance',         'score',       'Ability of government to formulate sound policies'),
('rule_of_law',                    'WGI',     'governance',         'score',       'Extent to which agents have confidence in and abide by rules'),
('voice_and_accountability',       'WGI',     'governance',         'score',       'Extent to which citizens participate in selecting government'),
-- IMF economic
('gdp_growth_pct',                 'IMF',     'economic',           'percent',     'Annual GDP growth rate'),
('inflation_pct',                  'IMF',     'economic',           'percent',     'Annual inflation rate'),
('unemployment_pct',               'IMF',     'economic',           'percent',     'Unemployment rate as % of total labor force'),
('current_account_balance_usd_bn', 'IMF',     'economic',           'billion USD', 'Current account balance in billions USD'),
('gross_govt_debt_pct_gdp',        'IMF',     'economic',           'percent',     'Gross government debt as % of GDP'),
('gdp_usd_bn',                     'IMF',     'economic',           'billion USD', 'GDP at current prices in billions USD'),
-- UNDP HDI
('hdi_value',                      'UNDP',    'human_development',  'index',       'Human Development Index (0–1)'),
('life_expectancy_years',          'UNDP',    'human_development',  'years',       'Life expectancy at birth'),
('expected_schooling_years',       'UNDP',    'human_development',  'years',       'Expected years of schooling'),
('mean_schooling_years',           'UNDP',    'human_development',  'years',       'Mean years of schooling'),
('gni_per_capita_2017ppp',         'UNDP',    'human_development',  '2017 PPP USD','GNI per capita in 2017 PPP USD'),
-- Polity5 democracy
('polity2_score',                  'Polity5', 'democracy',          'score',       'Combined democracy-autocracy score (-10 to +10)'),
('democracy_score',                'Polity5', 'democracy',          'score',       'Institutionalised democracy score (0–10)'),
('autocracy_score',                'Polity5', 'democracy',          'score',       'Institutionalised autocracy score (0–10)'),
-- V-Dem democracy
('electoral_democracy_index',      'V-Dem',   'democracy',          'index',       'Electoral democracy index (0–1)'),
('liberal_democracy_index',        'V-Dem',   'democracy',          'index',       'Liberal democracy index (0–1)'),
('participatory_democracy_index',  'V-Dem',   'democracy',          'index',       'Participatory democracy index (0–1)'),
-- Derived
('gdp_growth_yoy_calc',            'derived', 'economic',           'percent',     'Year-over-year GDP growth calculated from GDP levels'),
('governance_composite',           'derived', 'governance',         'score',       'Mean of available WGI scores per country-year'),
('regional_avg_gdp_growth',        'derived', 'economic',           'percent',     'Average GDP growth within region-year'),
('regional_avg_governance',        'derived', 'governance',         'score',       'Average governance composite within region-year')
ON CONFLICT (indicator_name, source) DO NOTHING;
