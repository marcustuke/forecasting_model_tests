import tensorflow as tf
import tensorflow_probability as tfp

trend = tfp.sts.LocalLinearTrend(observed_time_series=co2_by_month)
seasonal = tfp.sts.Seasonal(
    num_seasons=12, observed_time_series=co2_by_month)
model = tfp.sts.Sum([trend, seasonal], observed_time_series=co2_by_month)

from tensorflow_probability import distributions as tfd
from tensorflow_probability import sts

tf.enable_v2_behavior()

if tf.test.gpu_device_name() != "/device:GPU:0":
    print("WARNING: GPU device not found.")
else:
    print("SUCCESS: Found GPU: {}".format(tf.test.gpu_device_name()))

def tfp_sts_forecast(series, n_seasons=12, n_steps_forecast=20, param_samples=50, forecast_samples=10):
    trend = tfp.sts.LocalLinearTrend(observed_time_series=series)
    seasonal = tfp.sts.Seasonal(
        num_seasons=n_seasons, observed_time_series=series)

    model = tfp.sts.Sum([trend, seasonal], observed_time_series=series)
    variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model=model)

    forecast = tfp.sts.forecast(
        model, observed_time_series=series,
        num_steps_forecast=n_steps_forecast,
        parameter_samples=variational_posteriors.sample(param_samples)
    )

    return (
            forecast.mean()[..., 0],
            forecast.stddev()[..., 0],
            forecast.sample(forecast_samples)[..., 0],
    )
