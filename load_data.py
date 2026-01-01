{
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Load Results CSV\n",
                "This notebook loads the `results.csv` file using pandas and displays the first few rows."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "\n",
                "# Load the dataset\n",
                "df = pd.read_csv('results.csv')\n",
                "\n",
                "# Display the first 5 rows\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Display basic information about the dataset\n",
                "df.info()"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}