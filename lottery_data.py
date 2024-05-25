class LotteryData:
    type = "big"
    sorted_data = []
    sequence_data = []
    
    def __init__(self, type, sequence_data, sorted_data) -> None:
        self.type = type
        self.sequence_data = sequence_data
        self.sorted_data = sorted_data
    