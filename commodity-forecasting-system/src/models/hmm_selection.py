"""
HMM Model Selection Module
Optimal state selection using information criteria (AIC, BIC).
"""

import logging
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from src.models.hmm_core import CommodityHMM

logger = logging.getLogger(__name__)


class HMMModelSelector:
    """
    Model selection for HMM using information criteria.

    Information Criteria:
    - AIC (Akaike Information Criterion): 2k - 2ln(L)
    - BIC (Bayesian Information Criterion): k*ln(n) - 2ln(L)

    where:
    - k: number of parameters
    - n: number of observations
    - L: maximum likelihood

    BIC generally preferred for model selection as it imposes
    stronger penalty for model complexity.
    """

    def __init__(self, config: Dict):
        """
        Initialize model selector.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.results = []

    def select_optimal_states(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        n_states_range: range = range(2, 7),
        n_inits: int = 5,
        criterion: str = 'bic'
    ) -> Tuple[CommodityHMM, int, pd.DataFrame]:
        """
        Select optimal number of states using information criteria.

        Args:
            features: Feature matrix
            n_states_range: Range of states to test
            n_inits: Number of random initializations per model
            criterion: Selection criterion ('aic' or 'bic')

        Returns:
            Tuple of (best_model, best_n_states, results_df)
        """
        logger.info(
            f"Testing HMM with {len(n_states_range)} different state configurations"
        )

        # Convert to numpy if needed
        if isinstance(features, pd.DataFrame):
            features_array = features.values
        else:
            features_array = features

        n_samples, n_features = features_array.shape
        self.results = []

        for n_states in n_states_range:
            logger.info(f"Evaluating n_states={n_states}...")

            try:
                # Create temporary config with this n_states
                temp_config = self.config.copy()
                temp_config['hmm']['n_states'] = n_states

                # Fit model
                model = CommodityHMM(temp_config)
                model.fit_with_multiple_inits(features, n_inits=n_inits)

                # Calculate log-likelihood
                features_scaled = model.scaler.transform(features_array)
                log_likelihood = model.model.score(features_scaled) * n_samples

                # Calculate number of parameters
                # Transition matrix: n_states * (n_states - 1)
                # Initial state: n_states - 1
                # Emission means: n_states * n_features
                # Emission covariances: depends on covariance_type
                if model.covariance_type == 'diag':
                    cov_params = n_states * n_features
                elif model.covariance_type == 'full':
                    cov_params = n_states * n_features * (n_features + 1) // 2
                elif model.covariance_type == 'spherical':
                    cov_params = n_states
                elif model.covariance_type == 'tied':
                    cov_params = n_features * (n_features + 1) // 2
                else:
                    cov_params = n_states * n_features  # Default to diag

                n_params = (
                    n_states * (n_states - 1) +  # Transition matrix
                    (n_states - 1) +              # Initial state
                    n_states * n_features +       # Means
                    cov_params                    # Covariances
                )

                # Calculate AIC and BIC
                aic = 2 * n_params - 2 * log_likelihood
                bic = np.log(n_samples) * n_params - 2 * log_likelihood

                # Store results
                self.results.append({
                    'n_states': n_states,
                    'log_likelihood': log_likelihood,
                    'n_params': n_params,
                    'aic': aic,
                    'bic': bic,
                    'model': model
                })

                logger.info(
                    f"  n_states={n_states}: "
                    f"LL={log_likelihood:.2f}, "
                    f"AIC={aic:.2f}, "
                    f"BIC={bic:.2f}"
                )

            except Exception as e:
                logger.error(
                    f"TFM4001 INFERENCE: Failed to fit model with n_states={n_states}: {e}"
                )
                continue

        if not self.results:
            raise RuntimeError(
                "TFM4001 INFERENCE: Model selection failed for all configurations"
            )

        # Convert to DataFrame
        results_df = pd.DataFrame([
            {k: v for k, v in r.items() if k != 'model'}
            for r in self.results
        ])

        # Select best model
        if criterion == 'aic':
            best_idx = results_df['aic'].idxmin()
            logger.info(f"Selecting model with minimum AIC")
        elif criterion == 'bic':
            best_idx = results_df['bic'].idxmin()
            logger.info(f"Selecting model with minimum BIC")
        else:
            raise ValueError(
                f"TFM1001 CONFIG: Unknown criterion: {criterion}. "
                f"Use 'aic' or 'bic'."
            )

        best_result = self.results[best_idx]
        best_model = best_result['model']
        best_n_states = best_result['n_states']

        logger.info(
            f"\nOptimal model selected: n_states={best_n_states}"
        )
        logger.info(f"  Log-Likelihood: {best_result['log_likelihood']:.2f}")
        logger.info(f"  AIC: {best_result['aic']:.2f}")
        logger.info(f"  BIC: {best_result['bic']:.2f}")

        return best_model, best_n_states, results_df

    def plot_selection_criteria(
        self,
        results_df: pd.DataFrame,
        save_path: str = None
    ):
        """
        Plot AIC and BIC across different number of states.

        Args:
            results_df: Results DataFrame from select_optimal_states
            save_path: Optional path to save figure
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('AIC vs Number of States', 'BIC vs Number of States'),
            vertical_spacing=0.12
        )

        # AIC plot
        fig.add_trace(
            go.Scatter(
                x=results_df['n_states'],
                y=results_df['aic'],
                mode='lines+markers',
                name='AIC',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ),
            row=1, col=1
        )

        # BIC plot
        fig.add_trace(
            go.Scatter(
                x=results_df['n_states'],
                y=results_df['bic'],
                mode='lines+markers',
                name='BIC',
                line=dict(color='red', width=2),
                marker=dict(size=8)
            ),
            row=2, col=1
        )

        # Mark optimal points
        aic_min_idx = results_df['aic'].idxmin()
        bic_min_idx = results_df['bic'].idxmin()

        fig.add_trace(
            go.Scatter(
                x=[results_df.loc[aic_min_idx, 'n_states']],
                y=[results_df.loc[aic_min_idx, 'aic']],
                mode='markers',
                name='Optimal (AIC)',
                marker=dict(size=15, color='blue', symbol='star'),
                showlegend=True
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=[results_df.loc[bic_min_idx, 'n_states']],
                y=[results_df.loc[bic_min_idx, 'bic']],
                mode='markers',
                name='Optimal (BIC)',
                marker=dict(size=15, color='red', symbol='star'),
                showlegend=True
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_xaxes(title_text="Number of States", row=1, col=1)
        fig.update_xaxes(title_text="Number of States", row=2, col=1)
        fig.update_yaxes(title_text="AIC", row=1, col=1)
        fig.update_yaxes(title_text="BIC", row=2, col=1)

        fig.update_layout(
            title='HMM Model Selection: Information Criteria',
            height=700,
            showlegend=True
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Selection criteria plot saved to {save_path}")

        return fig


def compare_covariance_types(
    features: Union[pd.DataFrame, np.ndarray],
    config: Dict,
    n_states: int = 3,
    n_inits: int = 5
) -> pd.DataFrame:
    """
    Compare HMM performance across different covariance types.

    Args:
        features: Feature matrix
        config: Configuration dictionary
        n_states: Number of states to use
        n_inits: Number of random initializations

    Returns:
        DataFrame with comparison results
    """
    logger.info(f"Comparing covariance types for n_states={n_states}")

    covariance_types = ['diag', 'full', 'spherical', 'tied']
    results = []

    # Convert to numpy if needed
    if isinstance(features, pd.DataFrame):
        features_array = features.values
    else:
        features_array = features

    n_samples = len(features_array)

    for cov_type in covariance_types:
        logger.info(f"Testing covariance_type='{cov_type}'...")

        try:
            # Create temp config
            temp_config = config.copy()
            temp_config['hmm']['n_states'] = n_states
            temp_config['hmm']['covariance_type'] = cov_type

            # Fit model
            model = CommodityHMM(temp_config)
            model.fit_with_multiple_inits(features, n_inits=n_inits)

            # Calculate metrics
            features_scaled = model.scaler.transform(features_array)
            log_likelihood = model.model.score(features_scaled) * n_samples

            # Get convergence info
            converged = model.model.monitor_.iter < model.n_iter
            n_iter_used = model.model.monitor_.iter

            results.append({
                'covariance_type': cov_type,
                'log_likelihood': log_likelihood,
                'converged': converged,
                'n_iter': n_iter_used
            })

            logger.info(
                f"  {cov_type}: LL={log_likelihood:.2f}, "
                f"converged={converged}, iter={n_iter_used}"
            )

        except Exception as e:
            logger.error(
                f"TFM4001 INFERENCE: Failed with covariance_type='{cov_type}': {e}"
            )
            results.append({
                'covariance_type': cov_type,
                'log_likelihood': np.nan,
                'converged': False,
                'n_iter': np.nan
            })

    results_df = pd.DataFrame(results)
    return results_df


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('../..')

    from src.config.loader import load_config
    from src.config.logging_config import setup_logging
    from src.data.acquisition import CommodityDataAcquisition
    from src.data.preprocessing import DataPreprocessor
    from src.data.features import FeatureEngineer

    # Load configuration
    config = load_config('../../config/parameters.toml')
    logger = setup_logging(config)

    # Fetch and preprocess data
    data_client = CommodityDataAcquisition(config)
    data = data_client.fetch_commodity_prices()

    preprocessor = DataPreprocessor(config)
    data, _ = preprocessor.preprocess(data)

    # Engineer features
    feature_engineer = FeatureEngineer(config)
    features = feature_engineer.engineer_features(data)

    # Select subset of features
    feature_cols = ['returns', 'volatility_5', 'rsi_14', 'macd']
    hmm_features = features[feature_cols].dropna()

    print(f"\nPerforming model selection on {len(hmm_features)} samples")

    # Model selection
    selector = HMMModelSelector(config)
    best_model, best_n_states, results = selector.select_optimal_states(
        hmm_features,
        n_states_range=range(2, 6),
        n_inits=5,
        criterion='bic'
    )

    print("\n" + "="*80)
    print("MODEL SELECTION RESULTS")
    print("="*80)
    print(results)

    print(f"\nOptimal number of states: {best_n_states}")

    # Compare covariance types
    print("\n" + "="*80)
    print("COVARIANCE TYPE COMPARISON")
    print("="*80)

    cov_results = compare_covariance_types(
        hmm_features,
        config,
        n_states=best_n_states,
        n_inits=3
    )
    print(cov_results)
