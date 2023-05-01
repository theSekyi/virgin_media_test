CONFIG = {
    "one_hot_encode_columns": ["Type", "Gender"],
    "label_encode_columns": ["Vaccinated", "Sterilized", "Color1", "Color2"],
    "ordinal_encode_columns": {
        "Health": ["Healthy", "Minor Injury", "Serious Injury"],
        "FurLength": ["Short", "Medium", "Long"],
        "MaturitySize": ["Small", "Medium", "Large"],
    },
    "count_encode_column": "Breed1",
    "target_column": "Adopted",
}
