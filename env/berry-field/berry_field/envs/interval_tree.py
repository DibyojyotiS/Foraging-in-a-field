class IntervalTree:
    def __init__(self, data, start, end):
        # The data is assumed to be of the following form: [start, end, id],
        # where both start and end are included in the interval
        self.data = data

        self.start = start
        self.end = end

        self.tree = self.create_tree(data, start, end)

    def create_tree(self, node_list, start, end):
        if len(node_list) == 0:
            return []

        center_value = int(round((end + start)/2))

        left = []
        right = []
        center = []

        for node in node_list:
            if node[1] < center_value:
                left.append(node)
            elif node[0] > center_value:
                right.append(node)
            else:
                center.append(node)

        current_node = [center_value, sorted(center, key=lambda l: l[0]), sorted(center, key=lambda l: l[1])]

        return [current_node,
                self.create_tree(left, start, center_value - 1),
                self.create_tree(right, center_value + 1, end)]

    def find(self, val, tree):
        result = []

        if len(tree) == 0:
            return tree

        current_node = tree[0]

        if val == current_node[0]:
            return [v[2] for v in current_node[1]]

        if val < current_node[0]:
            for v in current_node[1]:
                if v[0] <= val:
                    result.append(v[2])
                else:
                    break

            return result + self.find(val, tree[1])

        else:
            for v in current_node[2]:
                if v[1] >= val:
                    result.append(v[2])
                else:
                    break

            return result + self.find(val, tree[2])

    def find_intervals(self, val):
        return set(self.find(val, self.tree))


def main():
    data = [
        [20,400,'id01'],
        [30,300,'id02'],
        [500,700,'id03'],
        [1020,2400,'id04'],
        [29949, 35891,'id05'],
        [899999,900000,'id06'],
        [999000,999000,'id07']
    ]

    tree = IntervalTree(data, 0, 1000000)

    print(tree.find_intervals(900001))


if __name__ == '__main__':
    main()