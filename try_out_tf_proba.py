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

def tfp_sts_forecast(series, n_seasons, n_steps_forecast, param_samples, forecast_samples):
    trend = tfp.sts.LocalLinearTrend(observed_time_series=co2_by_month)
    seasonal = tfp.sts.Seasonal(
        num_seasons=12, observed_time_series=co2_by_month)

    model = tfp.sts.Sum([trend, seasonal], observed_time_series=co2_by_month)
    variational_posteriors = tfp.sts.build_factored_surrogate_posterior(model=model)

    forecast = tfp.sts.forecast(
        model, observed_time_series=co2_by_month,
        num_steps_forecast=50,
        parameter_samples=variational_posteriors.sample(50)
    )

    return (
            forecast.mean()[..., 0],
            forecast.stddev()[..., 0],
            forecast.sample(10)[..., 0],
    )
