# Southern Maine Rental Price Predictor

This Streamlit prototype uses the final v2 joined rental and assessor dataset from the class project. It trains a regression model for monthly rent and a classifier for above-median rent when the app starts.

## Easiest way to run on Windows

1. Right-click the ZIP file and choose **Extract All**.
2. Open the extracted `some_rent_streamlit_app` folder.
3. Double-click `run_app.bat`.
4. Keep the black command window open while using the app.

The first launch may take a few minutes because it creates a local Python environment and installs the required packages. Later launches should be faster.

## Expected project folder layout

The app first looks for the latest class-project CSV in `SoME_outputs`. If it cannot find that file, it falls back to the packaged copy inside this app folder.

```text
project_folder/
  SoME_outputs/
    rental_assessor_town_join_latest.csv

  some_rent_streamlit_app/
    app.py
    model_utils.py
    run_app.bat
    requirements.txt
    rental_assessor_town_join_latest.csv
```

## Notes on the controls

Town and listing source are single-listing fields in the model. The app keeps them as single-select inputs for the main prediction, then provides comparison charts so users can compare the same assumptions across multiple towns or sources.

The feature-importance chart is global model importance. It does not change when a user changes the input fields because it describes which features the trained model used most overall.

## Prototype limitations

This app demonstrates the modeling workflow for the class project. It is not a production pricing tool. Better address-level matching, exact square footage, lease terms, utilities, parking, pet rules, and seasonality would make the model more useful in a real deployment.
