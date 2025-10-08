"""
Google Cloud Secret Manager integration module.
Retrieves credentials from Google Secret Manager using Application Default Credentials (ADC).
"""

import logging
from typing import Optional
from google.cloud import secretmanager
from google.api_core import exceptions as gcp_exceptions

logger = logging.getLogger(__name__)


class SecretManagerClient:
    """Client for retrieving secrets from Google Cloud Secret Manager."""

    def __init__(self, project_id: str):
        """
        Initialize Secret Manager client.

        Args:
            project_id: GCP project ID containing the secrets
        """
        self.project_id = project_id
        self.client = None

    def _get_client(self) -> secretmanager.SecretManagerServiceClient:
        """
        Get or create Secret Manager client.

        Returns:
            SecretManagerServiceClient instance

        Raises:
            Exception: If client initialization fails
        """
        if self.client is None:
            try:
                self.client = secretmanager.SecretManagerServiceClient()
                logger.info("Secret Manager client initialized successfully using ADC")
            except Exception as e:
                logger.error(f"Failed to initialize Secret Manager client: {e}")
                logger.error(
                    "Ensure GOOGLE_APPLICATION_CREDENTIALS is set or "
                    "you're running in an environment with ADC configured"
                )
                raise

        return self.client

    def get_secret(
        self,
        secret_name: str,
        version: str = "latest"
    ) -> Optional[str]:
        """
        Retrieve a secret value from Google Secret Manager.

        Args:
            secret_name: Name of the secret (e.g., 'reBOT_LOGIN')
            version: Version of the secret (default: 'latest')

        Returns:
            Secret value as string, or None if retrieval fails

        Raises:
            ValueError: If secret_name is empty
        """
        if not secret_name:
            raise ValueError("secret_name cannot be empty")

        try:
            client = self._get_client()

            # Build the resource name
            name = f"projects/{self.project_id}/secrets/{secret_name}/versions/{version}"

            logger.debug(f"Retrieving secret: {name}")

            # Access the secret version
            response = client.access_secret_version(request={"name": name})

            # Decode the secret payload
            secret_value = response.payload.data.decode('UTF-8')

            logger.info(f"Successfully retrieved secret: {secret_name}")
            return secret_value

        except gcp_exceptions.NotFound:
            logger.error(
                f"Secret not found: {secret_name} in project {self.project_id}. "
                f"Please verify the secret exists in Google Secret Manager."
            )
            return None

        except gcp_exceptions.PermissionDenied:
            logger.error(
                f"Permission denied accessing secret: {secret_name}. "
                f"Ensure the service account has 'Secret Manager Secret Accessor' role."
            )
            return None

        except Exception as e:
            logger.error(f"Error retrieving secret {secret_name}: {e}")
            return None

    def get_credentials(
        self,
        login_secret_name: str,
        password_secret_name: str
    ) -> Optional[dict]:
        """
        Retrieve both login ID and password from Secret Manager.

        Args:
            login_secret_name: Name of the login ID secret
            password_secret_name: Name of the password secret

        Returns:
            Dictionary with 'login_id' and 'password' keys, or None if retrieval fails
        """
        try:
            login_id = self.get_secret(login_secret_name)
            password = self.get_secret(password_secret_name)

            if login_id is None or password is None:
                logger.error("Failed to retrieve one or both credentials from Secret Manager")
                return None

            logger.info("Successfully retrieved both credentials from Secret Manager")

            return {
                'login_id': login_id,
                'password': password
            }

        except Exception as e:
            logger.error(f"Error retrieving credentials: {e}")
            return None

    def test_connection(self) -> bool:
        """
        Test connection to Secret Manager by listing secrets.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            client = self._get_client()
            parent = f"projects/{self.project_id}"

            # Try to list secrets (we don't need the results, just testing connectivity)
            request = secretmanager.ListSecretsRequest(parent=parent, page_size=1)
            client.list_secrets(request=request)

            logger.info("Secret Manager connection test successful")
            return True

        except gcp_exceptions.PermissionDenied:
            logger.error(
                "Permission denied. Service account needs 'Secret Manager Secret Accessor' role."
            )
            return False

        except Exception as e:
            logger.error(f"Secret Manager connection test failed: {e}")
            return False


def get_fmls_credentials(project_id: str, login_secret: str, password_secret: str) -> Optional[dict]:
    """
    Helper function to retrieve FMLS credentials from Secret Manager.

    Args:
        project_id: GCP project ID
        login_secret: Name of the login secret
        password_secret: Name of the password secret

    Returns:
        Dictionary with credentials, or None if retrieval fails
    """
    client = SecretManagerClient(project_id)
    return client.get_credentials(login_secret, password_secret)
