"""Per-dataset prompt content.

The role, domain context, guidance and feature descriptions that `prompts.py`
assembles into the system and user prompts, one entry per dataset.
"""
from __future__ import annotations

CONTENT = {
    'adult': {   'domain_context': [   'The model is trained on a subset of workers defined by employment '
                              'type.',
                              'Priors should emphasise stable socioeconomic relationships that '
                              'generalise across different labour market segments.'],
        'extra_notes': [   'Numeric variables stay in original units.',
                           'Categorical variables are represented as separate 0/1 one-hot '
                           'indicators with all categories retained.',
                           'workclass and sex are excluded from predictors in this locked split '
                           'setup.',
                           'occupation is grouped to top source-domain categories plus '
                           'Other_occupation to reduce sparse one-hot effects.'],
        'feature_descriptions': {   'age': 'age in years',
                                    'capital-gain': 'capital gains in dollars',
                                    'capital-loss': 'capital losses in dollars',
                                    'education-num': 'numeric education level',
                                    'fnlwgt': 'census final sample weight',
                                    'hours-per-week': 'usual work hours per week',
                                    'marital-status_Divorced': '0/1 indicator that marital status '
                                                               'is Divorced',
                                    'marital-status_Married-AF-spouse': '0/1 indicator that '
                                                                        'marital status is '
                                                                        'Married-AF-spouse',
                                    'marital-status_Married-civ-spouse': '0/1 indicator that '
                                                                         'marital status is '
                                                                         'Married-civ-spouse',
                                    'marital-status_Married-spouse-absent': '0/1 indicator that '
                                                                            'marital status is '
                                                                            'Married-spouse-absent',
                                    'marital-status_Never-married': '0/1 indicator that marital '
                                                                    'status is Never-married',
                                    'marital-status_Separated': '0/1 indicator that marital status '
                                                                'is Separated',
                                    'marital-status_Widowed': '0/1 indicator that marital status '
                                                              'is Widowed',
                                    'occupation_Adm-clerical': '0/1 indicator that occupation is '
                                                               'Adm-clerical',
                                    'occupation_Craft-repair': '0/1 indicator that occupation is '
                                                               'Craft-repair',
                                    'occupation_Exec-managerial': '0/1 indicator that occupation '
                                                                  'is Exec-managerial',
                                    'occupation_Other-service': '0/1 indicator that occupation is '
                                                                'Other-service',
                                    'occupation_Other_occupation': '0/1 indicator that occupation '
                                                                   'is Other_occupation',
                                    'occupation_Prof-specialty': '0/1 indicator that occupation is '
                                                                 'Prof-specialty',
                                    'occupation_Sales': '0/1 indicator that occupation is Sales'},
        'guidance': [   'Reflect stable socioeconomic signal directions.',
                        'Keep coefficient magnitudes realistic for a linear probability model.',
                        'One-hot indicator coefficients should usually be modest rather than '
                        'extreme.'],
        'outcome_desc': '- Linear probability model for income >50K: y in {0,1}, y = intercept + '
                        'sum_j beta_j * x_j',
        'role': 'a labor economist and applied statistician',
        'signal_hints': [   'Older age, more education, larger capital gains, and more work hours '
                            'often increase probability of income >50K.',
                            'marital-status_Married-civ-spouse is usually positive relative to '
                            'lower-income status categories.',
                            'Executive and professional occupation indicators are often positive, '
                            'while weaker labor-market categories are usually smaller or '
                            'negative.']},
    'boston_housing': {   'domain_context': [   'Source domain is younger neighbourhoods (proportion of '
                              'owner-occupied pre-1940 units AGE <= 70th percentile).',
                              'Priors should reflect structural urban-housing relationships that '
                              'hold across Boston-area neighbourhoods generally.'],
        'extra_notes': [   'B is a historically biased proxy capturing residential racial '
                           'composition; treat with caution and prefer small effects.',
                           'AGE is used only as the OOD splitter and is excluded from predictors '
                           '(matching the convention used for the other datasets where the '
                           'splitter is dropped).',
                           'CHAS is a binary 0/1 indicator; only ~7% of tracts bound the river so '
                           'its coefficient is on a low-prevalence indicator.',
                           'All other features are continuous in raw units.'],
        'feature_descriptions': {   'B': '1000(Bk - 0.63)^2 where Bk = proportion of residents of '
                                         'African-American descent (historically biased Boston '
                                         'housing covariate)',
                                    'CHAS': 'Charles River dummy (1 = tract bounds river, 0 = '
                                            'otherwise)',
                                    'CRIM': 'per-capita crime rate in the town',
                                    'DIS': 'weighted distance to five Boston employment centres',
                                    'INDUS': 'proportion of non-retail business acres per town',
                                    'LSTAT': 'percent of population with lower socioeconomic '
                                             'status',
                                    'NOX': 'nitric oxide concentration (parts per 10 million)',
                                    'PTRATIO': 'pupil-teacher ratio by town',
                                    'RAD': 'index of accessibility to radial highways',
                                    'RM': 'average number of rooms per dwelling',
                                    'TAX': 'full-value property tax rate per $10,000',
                                    'ZN': 'proportion of residential land zoned for lots over '
                                          '25,000 sq ft'},
        'guidance': [   'Use plausible signs grounded in urban-economics evidence.',
                        'Coefficients are in raw $1000 units of MEDV per one raw-unit change in '
                        'the predictor; keep magnitudes consistent with the feature scale '
                        'summaries.',
                        'Be conservative on tightly-correlated covariates (TAX/RAD, NOX/INDUS, '
                        'RM/LSTAT).'],
        'outcome_desc': '- Linear regression for median home value MEDV in $1000s: y = intercept + '
                        'sum_j beta_j * x_j',
        'role': 'an urban economist and applied statistician with experience in housing markets',
        'signal_hints': [   'More rooms per dwelling (RM) usually raise MEDV; lower-SES proportion '
                            '(LSTAT) usually lowers MEDV.',
                            'Higher crime (CRIM), pollution (NOX), and pupil-teacher ratio '
                            '(PTRATIO) typically lower MEDV.',
                            'Distance to employment centres (DIS) and highway accessibility (RAD) '
                            'trade off; expect modest, possibly opposing effects.',
                            'TAX rate effects are typically negative but small after conditioning '
                            'on RAD and INDUS.',
                            'ZN and CHAS effects are usually small; interpret with care given '
                            'low-prevalence categories.']},
    'cancer': {   'domain_context': [   'Source domain is U.S. South-region counties (AL, AR, DE, DC, FL, '
                              'GA, KY, LA, MD, MS, NC, OK, SC, TN, TX, VA, WV).',
                              'Priors should capture broad, transferable structural effects rather '
                              'than local artifacts.'],
        'extra_notes': [   'This is an ecological county-level dataset, so effects are '
                           'population-level associations rather than individual-level clinical '
                           'effects.',
                           'The active Cancer experiment uses a trimmed feature set designed to '
                           'reduce collinearity and make raw-space priors more stable.',
                           'Race-composition variables should be handled cautiously as structural '
                           'correlates, not causal or essentialist claims.'],
        'feature_descriptions': {   'MedianAge': 'median age',
                                    'PctBachDeg25_Over': "percent ages 25+ with a bachelor's "
                                                         'degree',
                                    'PctBlack': 'percent Black',
                                    'PctEmployed16_Over': 'employment rate among ages 16+',
                                    'PctHS25_Over': 'percent ages 25+ with high school as highest '
                                                    'education',
                                    'PctMarriedHouseholds': 'percent of households that are '
                                                            'married-couple households',
                                    'PctOtherRace': 'percent other race',
                                    'PctPrivateCoverage': 'percent with private health coverage',
                                    'PctPublicCoverage': 'percent with public health coverage',
                                    'PctUnemployed16_Over': 'unemployment rate among ages 16+',
                                    'incidenceRate': 'cancer incidence rate',
                                    'medIncome': 'median household income',
                                    'povertyPercent': 'percent of residents below the poverty '
                                                      'line'},
        'guidance': [   'Use plausible signs grounded in epidemiology.',
                        'Keep magnitudes realistic and conservative for county-level regressors.',
                        'Favor stable, interpretable structural effects over local artifacts.',
                        'Do not implicitly rely on excluded population-size proxies or redundant '
                        'count variables.'],
        'outcome_desc': '- County-level linear regression for TARGET_deathRate: y = intercept + '
                        'sum_j beta_j * x_j',
        'role': 'an epidemiologist specializing in population cancer outcomes, health disparities, '
                'and interpretable statistical modeling',
        'signal_hints': [   'Higher incidenceRate usually increases death rate.',
                            'Higher poverty, lower education, and lower employment often align '
                            'with worse mortality outcomes.',
                            'Higher private coverage and higher income are often protective, while '
                            'public-coverage variables can proxy older or more medically burdened '
                            'populations.',
                            'Because the retained socioeconomic and coverage variables are still '
                            'correlated, prefer moderate coefficients over aggressive values on '
                            'any single feature.']},
    'diabetes': {   'domain_context': [   'Source domain is younger patients (Age <= dataset median, computed '
                              'on the source-domain subset).',
                              'Priors should encode stable physiology associations likely to '
                              'transfer beyond the source age range.',
                              'Cohort is Pima Indians women aged 21+; the data was collected to '
                              'study type 2 diabetes onset.'],
        'extra_notes': [   'Age is used only for the OOD split definition and is excluded from '
                           'predictors.',
                           'Several features (Glucose, BloodPressure, SkinThickness, Insulin, BMI) '
                           'have observed zeros that are likely missing values; the pipeline '
                           'imputes them with the source-domain median.',
                           'DiabetesPedigreeFunction is a derived risk score capturing family '
                           'history of diabetes; it is unitless on a roughly 0-3 scale.'],
        'feature_descriptions': {   'BMI': 'body mass index (weight in kg / height in m^2)',
                                    'BloodPressure': 'diastolic blood pressure (mm Hg)',
                                    'DiabetesPedigreeFunction': 'diabetes pedigree function — a '
                                                                'score reflecting family history '
                                                                'of diabetes',
                                    'Glucose': 'plasma glucose concentration after 2-hour oral '
                                               'glucose tolerance test (mg/dL)',
                                    'Insulin': '2-hour serum insulin (μU/mL)',
                                    'Pregnancies': 'number of times pregnant',
                                    'SkinThickness': 'triceps skin fold thickness (mm)'},
        'guidance': [   'Use signs and magnitudes consistent with established type 2 diabetes '
                        'risk-factor evidence.',
                        'Prefer conservative values when uncertainty is high.',
                        'Avoid extreme magnitudes unless there is a strong physiological '
                        'rationale.'],
        'outcome_desc': '- Binary logistic regression for the positive class in the Pima Indians '
                        'Diabetes outcome (Outcome=1 = patient developed diabetes within 5 years '
                        'of the visit): logit(P(y=1)) = intercept + sum_j beta_j * x_j',
        'role': 'an endocrinologist and applied statistician with experience modelling type 2 '
                'diabetes risk in screening data',
        'signal_hints': [   'Higher Glucose is the strongest single predictor; expect a clear '
                            'positive coefficient.',
                            'Higher BMI raises diabetes risk; the coefficient should be positive '
                            "but smaller than Glucose's.",
                            'Higher Pregnancies and DiabetesPedigreeFunction tend to increase risk '
                            'modestly.',
                            'BloodPressure, SkinThickness, and Insulin effects are typically '
                            'smaller and noisier; coefficients can be near-zero.']},
    'heart_disease': {   'domain_context': [   'Source cohort is Cleveland.',
                              'Priors should emphasize clinically grounded effects likely to '
                              'remain stable outside the source cohort.'],
        'extra_notes': [   'Coded categorical variables are expanded to one-hot 0/1 indicators in '
                           'the design matrix.',
                           'Numeric variables (age, trestbps, chol, thalach, oldpeak) remain in '
                           'raw units.',
                           'Missing categorical values are filled with source-domain mode before '
                           'one-hot expansion.'],
        'feature_descriptions': {   'age': 'age in years',
                                    'cat__cp=1': '0/1 indicator that cp=typical_angina (chest pain '
                                                 'type coded 1=typical angina, 2=atypical angina, '
                                                 '3=non-anginal pain, 4=asymptomatic)',
                                    'cat__cp=2': '0/1 indicator that cp=atypical_angina (chest '
                                                 'pain type coded 1=typical angina, 2=atypical '
                                                 'angina, 3=non-anginal pain, 4=asymptomatic)',
                                    'cat__cp=3': '0/1 indicator that cp=non_anginal_pain (chest '
                                                 'pain type coded 1=typical angina, 2=atypical '
                                                 'angina, 3=non-anginal pain, 4=asymptomatic)',
                                    'cat__cp=4': '0/1 indicator that cp=asymptomatic (chest pain '
                                                 'type coded 1=typical angina, 2=atypical angina, '
                                                 '3=non-anginal pain, 4=asymptomatic)',
                                    'cat__exang=0': '0/1 indicator that exang=no_exercise_angina '
                                                    '(exercise-induced angina coded 1=yes, 0=no)',
                                    'cat__exang=1': '0/1 indicator that exang=exercise_angina '
                                                    '(exercise-induced angina coded 1=yes, 0=no)',
                                    'cat__fbs=0': '0/1 indicator that fbs=fbs_not_high (fasting '
                                                  'blood sugar > 120 mg/dl coded 1=true, 0=false)',
                                    'cat__fbs=1': '0/1 indicator that fbs=fbs_high (fasting blood '
                                                  'sugar > 120 mg/dl coded 1=true, 0=false)',
                                    'cat__restecg=0': '0/1 indicator that restecg=normal (resting '
                                                      'ECG category coded 0=normal, 1=ST-T '
                                                      'abnormality, 2=left ventricular '
                                                      'hypertrophy)',
                                    'cat__restecg=1': '0/1 indicator that restecg=st_t_abnormality '
                                                      '(resting ECG category coded 0=normal, '
                                                      '1=ST-T abnormality, 2=left ventricular '
                                                      'hypertrophy)',
                                    'cat__restecg=2': '0/1 indicator that restecg=lv_hypertrophy '
                                                      '(resting ECG category coded 0=normal, '
                                                      '1=ST-T abnormality, 2=left ventricular '
                                                      'hypertrophy)',
                                    'cat__sex=0': '0/1 indicator that sex=female (sex coded '
                                                  '1=male, 0=female)',
                                    'cat__sex=1': '0/1 indicator that sex=male (sex coded 1=male, '
                                                  '0=female)',
                                    'chol': 'serum cholesterol in mg/dl',
                                    'oldpeak': 'ST depression induced by exercise relative to rest',
                                    'thalach': 'maximum heart rate achieved',
                                    'trestbps': 'resting blood pressure in mm Hg'},
        'guidance': [   'Use clinically grounded signs for known risk factors.',
                        'Use moderate magnitudes unless evidence is strong.',
                        'With one-hot categorical indicators, distribute effects across categories '
                        'conservatively.',
                        'Prefer stable, transferable effects over source-specific quirks.'],
        'outcome_desc': '- Binary logistic regression for heart disease: logit(P(y=1)) = intercept '
                        '+ sum_j beta_j * x_j',
        'role': 'a cardiologist and clinical epidemiologist with strong statistical modeling '
                'expertise',
        'signal_hints': [   'Older age, exercise-induced angina, more ST depression, and higher '
                            'vessel-count categories usually raise risk.',
                            'Higher maximum heart rate is often protective after conditioning on '
                            'the other predictors.',
                            'Asymptomatic chest pain and flat/downsloping ST-segment categories '
                            'often align with higher risk.']},
}
