# PheMuT sequence dataset report

This report describes the independent paper-informed sequence dataset.
The default model feature set excludes yield history and red/ripe counts to reduce reactive one-window lag behaviour.

## Quality checks

                               check status    value threshold                                                                                     note
                sequence_target_rows   PASS 280.0000       > 0                                                          Rows are plot x future horizon.
                             seasons   PASS   2.0000      >= 2                Both seasons should be represented, but models will be fit within season.
                         input_weeks   PASS   5.0000         5                               Paper-informed default uses the first five observed weeks.
                               plots   PASS  40.0000     >= 20                                           Plot-level sequences provide validation units.
                     target_horizons   PASS   4.0000      >= 3                             Future horizon outputs should exist after the input history.
     incomplete_input_sequence_share   PASS   0.0000         0                                            Each sequence should contain all input weeks.
                target_missing_share   PASS   0.0000         0                                                               Target yield availability.
yield_like_features_in_default_model   PASS   0.0000         0 Default model features should not contain current/previous yield or target-like columns.
         default_model_feature_count   PASS  98.0000     5-120               Feature count should stay compact enough for 40 plot sequences per season.
  default_feature_mean_missing_share   PASS   0.0009    < 0.25                                         Mean missingness across selected model features.

## Target summary

 season  target_horizon_index target_date  n_plots  total_target_yield_g  mean_plot_target_yield_g  median_plot_target_yield_g  zero_yield_plot_share  mean_delta_vs_last_observed_g
   2324                     1  2024-02-13       40               16988.0                   424.700                       442.0                  0.000                         29.925
   2324                     2  2024-02-20       40               11512.0                   287.800                       284.0                  0.000                       -106.975
   2324                     3  2024-02-27       40               15085.0                   377.125                       355.5                  0.000                        -17.650
   2425                     1  2025-02-12       40                 940.0                    23.500                        19.0                  0.275                         11.325
   2425                     2  2025-02-20       40                2256.0                    56.400                        45.0                  0.075                         44.225
   2425                     3  2025-02-26       40                1099.0                    27.475                        25.5                  0.150                         15.300
   2425                     4  2025-03-05       40                2126.0                    53.150                        42.5                  0.000                         40.975

## Feature sets

               feature_set  n_features
paper_informed_nonreactive          98
 paper_informed_no_weather          53
      weather_only_context          48
