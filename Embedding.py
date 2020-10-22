class Embedding:
    def __init__(self, values):
        self._value_to_id = dict()
        self._id_to_value = dict()
        self._generate_mapping(values)

    def _generate_mapping(self, values):
        for index, value in enumerate(values):
            id = index + 1
            self._value_to_id[value] = id
            self._id_to_value[id] = value

    def value_to_id(self, value):
        return self._value_to_id[value]

    def id_to_value(self, id):
        return self._id_to_value[id]

    def size(self):
        return len(self._id_to_value)
