# Southern Maine Rental Price Predictor

This Streamlit prototype uses the final v2 joined rental and assessor dataset from the class project. It trains a regression model for monthly rent and a classifier for above-median rent when the app starts.

The app is intended as a proof of concept for rental comparison. It is not a production pricing system.

## What the app does

The app allows a user to enter interpretable rental-listing features, including:

- town
- listing source
- bedrooms
- bathrooms
- square footage
- furnished status
- selected listing-text features, such as waterfront, parking, laundry, utilities, pet-friendly, and luxury wording

The app returns:

- a predicted monthly rent
- an above-median versus at/below-median classification
- comparison charts for similar assumptions across towns, sources, and bedroom counts
- global model feature importance

## Files in this repository

For Streamlit Community Cloud, the app files should live in the same repository folder:

```text
southern-maine-rent-predictor/
  app.py
  model_utils.py
  requirements.txt
  rental_assessor_town_join_latest.csv
  README.md
  run_app.bat
```

The `run_app.bat` file is only needed for running the app locally on Windows. Streamlit Community Cloud uses `app.py` as the main app file and installs packages from `requirements.txt`.

The dataset file, `rental_assessor_town_join_latest.csv`, is included with the app so the prototype can run without requiring the full class-project folder structure.

## Deploying on Streamlit Community Cloud

1. Upload the files above to a GitHub repository.
2. Go to Streamlit Community Cloud.
3. Create a new app from the GitHub repository.
4. Select the main branch.
5. Set the main file path to:

```text
app.py
```

6. Deploy the app.

The first launch may take a few minutes while Streamlit installs the required packages and trains the models.

## Easiest way to run locally on Windows

1. Download the ZIP file.
2. Right-click the ZIP file and choose **Extract All**.
3. Open the extracted app folder.
4. Double-click `run_app.bat`.
5. Keep the black command window open while using the app.

The first local launch may take a few minutes because it creates a local Python environment and installs the required packages. Later launches should be faster.

## Notes on the controls

Town and listing source are single-listing fields in the model. The app keeps them as single-select inputs for the main prediction, then provides comparison charts so users can compare the same listing assumptions across multiple towns or sources.

The feature-importance chart is global model importance. It does not change when a user changes the input fields because it describes which features the trained model used most overall.

## Prototype limitations

This app demonstrates the modeling workflow for the class project. It should be used as a comparison and screening tool, not as a final pricing authority.

Important limitations include:

- public listing data can be incomplete or inconsistent
- square footage is missing for many listings
- the model does not fully capture unit condition, lease terms, utilities, parking, pet rules, or seasonality
- town-level assessor context is useful, but it is not the same as exact address-level property matching
- the model should be monitored and retrained before any production use

Better address-level matching, stronger square-footage coverage, lease terms, utilities, parking, pet rules, and seasonality would make the model more useful in a real deployment.
