import utix.plotex as plotex
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_colwidth', None)
df = pd.read_csv('eval_result_159791128.csv')
plotex.pd_series_plot(df=df,
                      output_path='.',
                      group_cols=('group_type', 'test'),
                      groups=(
                          ('all', 'rewrite_in_history_data-2020_04_test_CNNfix_oldL2_201908_debug-test-2020_04_test_CNNfix_oldL2_201908-test_prod'),
                          ('Music', 'rewrite_in_history_data-2020_04_test_CNNfix_oldL2_201908_debug-test-2020_04_test_CNNfix_oldL2_201908-test_prod'),
                          ('HomeAutomation', 'rewrite_in_history_data-2020_04_test_CNNfix_oldL2_201908_debug-test-2020_04_test_CNNfix_oldL2_201908-test_prod')
                      ),
                      series_col='model',
                      index_col='threshold_pos',
                      value_cols=('precision', 'recall'),
                      xlabel='threshold',
                      plot_args={
                          'precision': {
                              'title': 'hist_p@1_est',
                              'marker': 'o'
                          },
                          'recall': {
                              'title': 'hist_trig_est',
                              'marker': '^',
                              'linestyle': 'dashed'
                          },
                      })
plt.show()
