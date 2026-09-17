# PheMuT temporal alignment audit

## Quality checks

```csv
check,status,value,threshold,note
plot_current_yield_lookup_missing_share,PASS,0.0,0,Every plot forecast origin should map to a current tidy observation.
plot_target_yield_lookup_missing_share,PASS,0.0,0,Every plot target date should map to a future tidy observation.
plot_current_yield_max_abs_diff_g,PASS,0.0,0,Feature current yield must equal tidy yield at forecast origin.
plot_target_yield_max_abs_diff_g,PASS,0.0,0,Target next_yield_g must equal tidy yield at target date.
plot_non_future_target_rows,PASS,0.0,0,All target dates must be after forecast origin dates.
plot_forecast_horizon_min_days,PASS,5.0,> 0,Forecast horizon should be strictly positive.
plot_forecast_horizon_max_days,PASS,9.0,<= 21,Long gaps should be reviewed.
aggregate_current_lookup_missing_share,PASS,0.0,0,Every aggregate forecast origin should map to current aggregate observations.
aggregate_target_lookup_missing_share,PASS,0.0,0,Every aggregate target date should map to future aggregate observations.
aggregate_current_yield_max_abs_diff_g,PASS,0.0,0,Aggregate current yield must equal sum of plot yields at forecast origin.
aggregate_target_yield_max_abs_diff_g,PASS,0.0,0,Aggregate target yield must equal sum of plot yields at target date.
aggregate_non_future_target_rows,PASS,0.0,0,All aggregate target dates must be after forecast origin dates.
aggregate_n_plot_mismatch_rows,PASS,0.0,0,Aggregate feature n_plots should equal tidy plot count at forecast origin.
future_target_features_selected,PASS,0.0,0,Selected features should not include target/next/future/error columns.
lag_warning_models,WARN,27.0,0,Number of model/task combinations with one-window lag warning.
non_persistence_lag_warning_models,WARN,21.0,0,Lag warnings on learned/non-persistence models are more concerning.
```

## Plot-level alignment sample

```csv
season,plot_id,observation_date,target_date,forecast_horizon_days,yield_g,next_yield_g,delta_next_yield_g,flower_count,green_fruit_count,white_fruit_count,pink_fruit_count,red_or_ripe_fruit_count,current_yield_diff_g,target_yield_diff_g
2425,B11,2025-01-07,2025-01-15,8.0,27.0,0.0,-27.0,0.0,3.0,1.0,1.0,3.0,0.0,0.0
2425,B11,2025-01-15,2025-01-24,9.0,0.0,8.0,8.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0
2425,B11,2025-02-05,2025-02-12,7.0,0.0,2.0,2.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0
2425,B11,2025-02-12,2025-02-20,8.0,2.0,0.0,-2.0,1.0,2.0,0.0,0.0,0.0,0.0,0.0
2425,B11,2025-02-20,2025-02-26,6.0,0.0,8.0,8.0,1.0,7.0,0.0,0.0,0.0,0.0,0.0
2425,B11,2025-02-26,2025-03-05,7.0,8.0,19.0,11.0,0.0,8.0,0.0,0.0,1.0,0.0,0.0
2425,B12,2025-01-07,2025-01-15,8.0,11.0,0.0,-11.0,0.0,1.0,2.0,0.0,2.0,0.0,0.0
2425,B12,2025-01-15,2025-01-24,9.0,0.0,3.0,3.0,0.0,8.0,1.0,0.0,0.0,0.0,0.0
2425,B12,2025-01-24,2025-01-29,5.0,3.0,5.0,2.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0
2425,B12,2025-01-29,2025-02-05,7.0,5.0,18.0,13.0,0.0,4.0,0.0,0.0,1.0,0.0,0.0
2425,B12,2025-02-05,2025-02-12,7.0,18.0,19.0,1.0,1.0,1.0,0.0,0.0,3.0,0.0,0.0
2425,B12,2025-02-12,2025-02-20,8.0,19.0,45.0,26.0,4.0,18.0,0.0,0.0,2.0,0.0,0.0
2425,B12,2025-02-20,2025-02-26,6.0,45.0,72.0,27.0,0.0,34.0,8.0,1.0,5.0,0.0,0.0
2425,B12,2025-02-26,2025-03-05,7.0,72.0,50.0,-22.0,2.0,10.0,4.0,1.0,5.0,0.0,0.0
2425,B13,2025-01-07,2025-01-15,8.0,47.0,0.0,-47.0,0.0,5.0,0.0,2.0,15.0,0.0,0.0
2425,B13,2025-01-15,2025-01-24,9.0,0.0,14.0,14.0,0.0,2.0,0.0,0.0,1.0,0.0,0.0
2425,B13,2025-01-24,2025-01-29,5.0,14.0,5.0,-9.0,0.0,9.0,1.0,0.0,1.0,0.0,0.0
2425,B13,2025-01-29,2025-02-05,7.0,5.0,17.0,12.0,0.0,2.0,0.0,1.0,0.0,0.0,0.0
2425,B13,2025-02-05,2025-02-12,7.0,17.0,10.0,-7.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
2425,B13,2025-02-12,2025-02-20,8.0,10.0,7.0,-3.0,3.0,8.0,0.0,0.0,0.0,0.0,0.0
```

## Aggregate alignment sample

```csv
season,observation_date,target_date,forecast_horizon_days,n_plots,aggregate_yield_g,aggregate_next_yield_g,aggregate_delta_next_yield_g,current_aggregate_diff_g,target_aggregate_diff_g,aggregate_flower_count,aggregate_green_fruit_count,aggregate_white_fruit_count,aggregate_pink_fruit_count,aggregate_red_or_ripe_fruit_count
2425,2025-01-07,2025-01-15,8.0,40,1210.0,417.0,-793.0,0.0,0.0,12.0,134.0,42.0,29.0,136.0
2425,2025-01-15,2025-01-24,9.0,40,417.0,101.0,-316.0,0.0,0.0,9.0,190.0,31.0,27.0,50.0
2425,2025-01-24,2025-01-29,5.0,40,101.0,66.0,-35.0,0.0,0.0,2.0,132.0,22.0,13.0,39.0
2425,2025-01-29,2025-02-05,7.0,40,66.0,487.0,421.0,0.0,0.0,2.0,137.0,15.0,12.0,33.0
2425,2025-02-05,2025-02-12,7.0,40,487.0,940.0,453.0,0.0,0.0,9.0,185.0,43.0,7.0,46.0
2425,2025-02-12,2025-02-20,8.0,40,940.0,2256.0,1316.0,0.0,0.0,191.0,665.0,79.0,38.0,105.0
2425,2025-02-20,2025-02-26,6.0,40,2256.0,1099.0,-1157.0,0.0,0.0,22.0,577.0,96.0,31.0,154.0
2425,2025-02-26,2025-03-05,7.0,40,1099.0,2126.0,1027.0,0.0,0.0,37.0,497.0,86.0,25.0,82.0
```

## Lag/timeliness audit sample

```csv
task_id,task_label,level,target_mode,model_id,model_label,n_target_windows,same_window_corr,forecast_vs_current_window_corr,forecast_vs_previous_observed_target_corr,same_window_mae_g,forecast_vs_current_window_mae_g,forecast_vs_previous_observed_target_mae_g,direction_accuracy_vs_current,lag_warning,lag_note
plot_delta_current_state,Plot-level delta-to-next · current-state model,plot,delta,persistence,Persistence / no-change baseline,8,0.42178392290118005,0.9999999999999999,0.9999933300111018,688.75,0.0,1.1428571428571428,0.0,True,forecast aligns more with current window than target; forecast is closer to current window than target; forecast aligns more with previous target than target; persistence lag is expected by definition
plot_next_current_state,Plot-level next yield · current-state upper bound,plot,next,persistence,Persistence / no-change baseline,8,0.42178392290118005,0.9999999999999999,0.9999933300111018,688.75,0.0,1.1428571428571428,0.0,True,forecast aligns more with current window than target; forecast is closer to current window than target; forecast aligns more with previous target than target; persistence lag is expected by definition
plot_next_leading,Plot-level next yield · leading indicators,plot,next,persistence,Persistence / no-change baseline,8,0.42178392290118005,0.9999999999999999,0.9999933300111018,688.75,0.0,1.1428571428571428,0.0,True,forecast aligns more with current window than target; forecast is closer to current window than target; forecast aligns more with previous target than target; persistence lag is expected by definition
aggregate_delta_current_state,Direct aggregate delta-to-next · current-state model,aggregate,delta,persistence,Persistence / no-change baseline,8,0.4208412536053367,1.0,1.0,689.75,0.0,0.0,0.0,True,forecast aligns more with current window than target; forecast is closer to current window than target; forecast aligns more with previous target than target; persistence lag is expected by definition
aggregate_next_current_state,Direct aggregate next window · current-state upper bound,aggregate,next,persistence,Persistence / no-change baseline,8,0.4208412536053367,1.0,1.0,689.75,0.0,0.0,0.0,True,forecast aligns more with current window than target; forecast is closer to current window than target; forecast aligns more with previous target than target; persistence lag is expected by definition
aggregate_next_leading,Direct aggregate next window · leading indicators,aggregate,next,persistence,Persistence / no-change baseline,8,0.4208412536053367,1.0,1.0,689.75,0.0,0.0,0.0,True,forecast aligns more with current window than target; forecast is closer to current window than target; forecast aligns more with previous target than target; persistence lag is expected by definition
plot_delta_current_state,Plot-level delta-to-next · current-state model,plot,delta,extra_trees,Extra Trees,8,-0.14896337571981394,0.4403828975540379,0.7305958972889695,1978.3475896700663,1912.9019482372692,1247.265678306343,0.5,True,forecast aligns more with current window than target; forecast aligns more with previous target than target
plot_next_leading,Plot-level next yield · leading indicators,plot,next,random_forest,Random Forest,8,-0.2132093758797712,0.1531418130885282,-0.005261187388923643,2162.3995524672328,2088.3177704840737,1596.6236477976843,0.5,True,forecast aligns more with current window than target; forecast aligns more with previous target than target
plot_delta_current_state,Plot-level delta-to-next · current-state model,plot,delta,gradient_boosting,Gradient Boosting,8,0.033566660575256785,0.21529431574466157,0.13462204348781714,2213.5781358345403,2335.8281358345403,2087.5060832489166,0.625,True,forecast aligns more with current window than target; forecast aligns more with previous target than target
plot_next_current_state,Plot-level next yield · current-state upper bound,plot,next,random_forest,Random Forest,8,-0.2683061008170388,0.0776567108780402,-0.05141065149680553,2230.4513472601275,2046.4374871141238,1711.5212287790255,0.5,True,forecast aligns more with current window than target; forecast aligns more with previous target than target
plot_next_current_state,Plot-level next yield · current-state upper bound,plot,next,extra_trees,Extra Trees,8,-0.40094684552414844,0.2120730853622776,0.05405426777938117,2245.2085486719316,2231.6080348783053,1798.0074079215824,0.5,True,forecast aligns more with current window than target; forecast aligns more with previous target than target
plot_next_leading,Plot-level next yield · leading indicators,plot,next,gradient_boosting,Gradient Boosting,8,-0.11930872413293109,0.22032511143752284,0.09015425525057785,2314.8377459489348,2369.898956921174,1798.8183076896328,0.5,True,forecast aligns more with current window than target; forecast aligns more with previous target than target
plot_next_leading,Plot-level next yield · leading indicators,plot,next,extra_trees,Extra Trees,8,-0.4455628339368736,0.10691681994887549,-0.1340994751039821,2374.3062755155024,2366.882476227139,1955.5297412018429,0.5,True,forecast aligns more with current window than target; forecast aligns more with previous target than target
plot_next_current_state,Plot-level next yield · current-state upper bound,plot,next,gradient_boosting,Gradient Boosting,8,-0.2667418376492393,0.1905354894921847,0.03376945429330243,2431.9653017805012,2220.153152208033,1596.7487220526848,0.5,True,forecast aligns more with current window than target; forecast aligns more with previous target than target
plot_delta_current_state,Plot-level delta-to-next · current-state model,plot,delta,random_forest,Random Forest,8,-0.09599846459026352,0.30324262117896184,0.21679170289927888,2473.231005009151,2373.275542521983,1852.0513607896717,0.625,True,forecast aligns more with current window than target; forecast aligns more with previous target than target
plot_delta_current_state,Plot-level delta-to-next · current-state model,plot,delta,elasticnet,ElasticNet regression,8,0.14695212922154902,0.39474772569662037,0.3918135310368217,2867.53802860631,2741.4760962929995,1687.4009295603278,0.625,True,forecast aligns more with current window than target; forecast aligns more with previous target than target
plot_next_current_state,Plot-level next yield · current-state upper bound,plot,next,elasticnet,ElasticNet regression,8,0.16935016776723705,0.40858633837404434,0.4113259015323863,2970.9645848595414,2879.742563907043,1805.9410500174288,0.625,True,forecast aligns more with current window than target; forecast aligns more with previous target than target
plot_delta_current_state,Plot-level delta-to-next · current-state model,plot,delta,ridge,Ridge regression,8,0.20684511441833273,0.3998256584214966,0.387035804960408,3245.701119526068,3220.0806451906974,2085.5119135918726,0.75,True,forecast aligns more with current window than target; forecast aligns more with previous target than target
plot_next_current_state,Plot-level next yield · current-state upper bound,plot,next,ridge,Ridge regression,8,0.2155928263085681,0.40656362472076546,0.3968445984989986,3321.7446733369943,3321.215488943484,2170.477049793448,0.75,True,forecast aligns more with current window than target; forecast aligns more with previous target than target
plot_next_leading,Plot-level next yield · leading indicators,plot,next,elasticnet,ElasticNet regression,8,0.2575552431307215,0.4635133171815725,0.47528405306351973,3465.8973781489435,3509.6462945980384,2397.683406569414,0.75,True,forecast aligns more with current window than target; forecast aligns more with previous target than target
```
