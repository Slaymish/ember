import ember

def ember_to_vector(train_path, test_path, output_dir):
    ember.create_vectorized_features(output_dir, feature_version=2, train_feature_paths=[train_path], test_feature_paths=[test_path])


def main():
    train_dst = "data/ember/train.jsonl"
    test_dst = "data/ember/test.jsonl"
    # def create_vectorized_features(output_dir, feature_version=2, train_feature_paths=None, test_feature_paths=None):
    ember.create_vectorized_features("data/vectors", feature_version=2, train_feature_paths=[train_dst], test_feature_paths=[test_dst])

if __name__ == "__main__":
    main()
