from pipelines.training_pipeline import training_pipeline

# Add the root directory to the Python path
if __name__ == "__main__":
    # Run the pipeline
    training_pipeline(
        data_path=(
            "/Users/georgensamuel/Documents/Machine_Learning_Projects/customer-satisfaction-zenml/data/"
            "olist_customers_dataset.csv"
        )
    )
