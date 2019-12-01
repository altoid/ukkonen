import logging
import sys
import unittest


def get_logger():
    logger = logging.getLogger('ukkonen')

    formatter = logging.Formatter('%(asctime)s - %(filename)s:%(funcName)s:%(lineno)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    level = logging.DEBUG
    logger.setLevel(level)
    ch.setLevel(level)

    return logger


logger = get_logger()


class Node(object):
    # edges are tuples (index, length) which are used as
    # keys in a dictionary.  the values are other nodes.

    def __init__(self, tree):
        self.tree = tree
        self.suffix_link = None
        self.children = {}
        self.parent = None

    def add_node(self, arc):
        # an arc is a 2-ple (left, right) which gives the index of the first and last char of the arc label.
        # this must specify at least one character, and the indices cannot run past the end of the text.

        assert(arc not in self.children)
        assert(0 <= arc[0] <= arc[1] < len(self.tree.text))

        n = Node(self.tree)
        self.children[arc] = n
        n.parent = self
        return self.children[arc]

    def is_leaf(self):
        return len(self.children) == 0

    def arc_exists(self, c):
        for k in self.children.keys():
            text = self.tree.text
            if c == text[k[0]]:
                return True
        return False

    def match_helper(self, p):
        if len(p) == 0:
            return True

        # go through all the children to see if any of them match
        for k in self.children.keys():
            x = p[:k[1]]
            y = self.tree.text[k[0]:k[0] + k[1]]
            if x == y:
                nextnode = self.children[k]
                return nextnode.match_helper(p[k[1]:])
        return False

    def show(self, **kwargs):
        level = kwargs.get('level', 0)
        if self.suffix_link:
            print "%s%s (suffix link:  %s)" % ('    ' * level, id(self), id(self.suffix_link))
        else:
            print "%s%s" % ('    ' * level, id(self))
        for k in self.children.keys():
            print '%s%s - %s' % ('    ' * (level + 1), k, self.tree.get_arc_label(k))
            self.children[k].show(level=level + 2)

    def get_key(self, c):
        """
        :param c:
        :return:  the 2-ple corresponding to the arc that starts with c, or None
        """
        for k in self.children.keys():
            if c == self.tree.text[k[0]]:
                return k

    def split_arc(self, arc, n):
        """
        split this arc before the nth character on its edge label.

        0 < n < length of arc label

        this only splits the arc and creates a new internal node.  does not create leaves.

        split(root, (0, 5), 1) does this:

            a  x  a  b  x  $
        o -------------------> *

            a      x  a  b  x  $
        o ---- * ----------------> *

        :param arc:
        :param n:
        :return: new internal node created by the split operation.
        """

        assert(0 < n < arc[1] - arc[0] + 1)  # this forces arc label to be at least 2 chars.
        assert(arc in self.children)

        child = self.children[arc]
        new_node = Node(self.tree)
        del self.children[arc]
        self.children[(arc[0], arc[0] + n - 1)] = new_node
        new_node.children[(arc[0] + n, arc[1])] = child
        child.parent = new_node
        new_node.parent = self
        return new_node

    def traverse(self, s):
        """
        beginning from node, traverse from it on the string s.

        :param node:
        :param s:
        :return: if s ends at a node, return that node.
        if s ends in the middle of an arc label, return the node from which that arc emerges.
        if s cannot be matched by traversing from node, return None.
        """

        assert(s is not None)
        if len(s) == 0:
            return self

        text = self.tree.text

        # find which arc to use
        c = s[0]
        arc = None
        for k in self.children.keys():
            if c == text[k[0]]:
                arc = k
                break

        if not arc:
            return None

        arc_len = arc[1] - arc[0] + 1
        chars_to_match = min(arc_len, len(s))
        if s[:chars_to_match] != text[arc[0]:(arc[0] + chars_to_match)]:
            return None

        # ok, we have a match.

        # there is more of s to match.  keep going.
        child = self.children[arc]
        if len(s) > arc_len:
            return child.traverse(s[arc_len:])

        if len(s) == arc_len:
            return child

        return self

    def get_arc_from_parent(self):
        # TODO:  devise a better way to do this.  e.g. change traverse to return both the node and the arc
        # TODO:  from the parent
        if self.parent is None:
            return None

        for i in self.parent.children.items():
            if i[1] == self:
                return i[0]


class SuffixTree(object):
    def __init__(self, text):
        self.__text = text + '$'
        self.root = Node(self)
        self.leaves = []

    def append_helper(self, node):
        for t in node.children.keys():
            child = node.children[t]
            if len(child.children) == 0:
                del node.children[t]
                node.children[(t[0], t[1] + 1)] = child
                child.parent = node
            else:
                self.append_helper(child)

    def get_arc_label(self, arc):
        return self.text[arc[0]:arc[1] + 1]

    def show(self):
        print self.text
        self.root.show()

    def append_to_leaves(self):
        # this will just increment the lengths for arcs pointing to leaf nodes
        self.append_helper(self.root)

    @property
    def text(self):
        return self.__text

    def matches(self, p):
        return self.root.match_helper(p)

    def text_fragment(self, arc):
        return self.text[arc[0]:(arc[1] + 1)]

    def is_substring(self, s):
        # current_node = self.root
        # i = 0
        # t = current_node.get_key(s[i])
        # while t:
        #     min_l = min(t[1], len(s[i:]))
        #     if self.text[t[0]:t[0] + min_l] != s[i:i + min_l]:
        #         return False
        #
        #     i += min_l
        #     if i == len(s):
        #         return True
        #
        #     current_node = current_node.children[t]
        #     t = current_node.get_key(s[i])
        #
        return False

    def is_suffix(self, s):
        return self.is_substring(s + '$')

    def count_suffixes(self):
        return self.root.count_suffixes()

    def dump_suffix_helper(self, partial_string, node):
        if node.is_leaf():
            yield partial_string
            return

        for k, v in node.children.items():
            for x in self.dump_suffix_helper(partial_string + self.text_fragment(k), v):
                yield x

    def suffixes(self):
        return self.dump_suffix_helper('', self.root)

    def add_node(self, node, arc):
        assert(node.tree == self)
        assert(arc not in node.children)
        new_node = node.add_node(arc)
        self.leaves.append(new_node)

    def traverse(self, node, s):
        assert(s is not None)
        assert(node.tree == self)

        return node.traverse(s)

    def partial_path(self, origin, terminus):
        """
        return the concatenation of the arc labels from origin to terminus.
        :param origin:
        :param terminus:
        :return: returns empty string if origin == terminus.  raises exception if terminus cannot be reached from
        origin
        """
        result = ""
        p = terminus
        while p:
            if p == origin:
                return result
            a = p.get_arc_from_parent()
            result = self.get_arc_label(a) + result
            p = p.parent
        raise Exception("partial_path - origin does not precede terminus")

    def build_tree(self):
        # add the first character to construct T1.

        # phase and extension numbers are 1-based.

        logger.debug("phase 1 - constructing T1")
        self.add_node(self.root, (0, 0))

        m = len(self.text)
        i = 1
        nleaves = 1
        while i < m:
            current_char = self.text[i]

            nleaves = len(self.leaves)
            logger.debug("phase i + 1 = %s" % (i + 1))
            logger.debug("adding %s to each of the %s leaves" % (current_char, nleaves))
            self.append_to_leaves()

            # after we extend all the leaves, start traversing from the root.
            current_node = self.root
            last_internal_node = None

            j = nleaves + 1
            while j < i + 2:
                logger.debug("extension j = %s" % j)

                prev_extension_suffix = self.text[j - 2:i]
                cur_extension_suffix = prev_extension_suffix[1:]

                logger.debug("prev suffix = %s, current suffix = %s" % (prev_extension_suffix, cur_extension_suffix))

                v = self.traverse(current_node, prev_extension_suffix)
                if v == self.root:
                    # need to ensure that cur_extension_suffix + current_char is in the tree.  figure out which
                    # extension rule applies.
                    new_internal_node = self.apply_extension_rules(cur_extension_suffix, current_char, i, v)
                else:
                    sv = v.suffix_link
                    if sv == self.root:
                        new_internal_node = self.apply_extension_rules(cur_extension_suffix, current_char, i, sv)
                    else:
                        partial_path = self.partial_path(current_node, v)
                        assert(prev_extension_suffix.startswith(partial_path))
                        gamma = prev_extension_suffix[len(partial_path):]
                        new_internal_node = self.apply_extension_rules(gamma, current_char, i, sv)

                if new_internal_node:
                    if last_internal_node:
                        last_internal_node.suffix_link = new_internal_node
                    new_internal_node.suffix_link = self.root
                    last_internal_node = new_internal_node
                j += 1
            i += 1
        logger.debug("suffix tree complete; %s leaves" % len(self.leaves))

    def apply_extension_rules(self, cur_extension_suffix, current_char, i, v):
        """

        :param cur_extension_suffix:
        :param current_char:
        :param i:
        :param v: node from which w
        :return: if a new internal node is created, return it.  otherwise return None
        """
        vv = self.traverse(v, cur_extension_suffix)
        if vv != v:
            # find the concatenation of the labels between v and vv
            partial_path = self.partial_path(v, vv)
            assert(cur_extension_suffix.startswith(partial_path))
            cur_extension_suffix = cur_extension_suffix[len(partial_path):]

        if not cur_extension_suffix:
            arc = vv.get_key(current_char)
            if arc is not None:
                # cur_extension_suffix + current_char is already here:  rule 3
                pass
            else:
                # have to add new edge
                self.add_node(vv, (i, i))
        else:
            # find the end of cur_extension_suffix.
            arc = vv.get_key(cur_extension_suffix[0])
            label = self.get_arc_label(arc)
            # since vv is root, we know that cur_extension_suffix does not end at another node.  therefore
            # the label is at least 1 char longer than cur_extension_suffix.
            if label.startswith(cur_extension_suffix + current_char):
                # cur_extension_suffix + current_char is already here:  rule 3
                pass
            else:
                # have to split
                new_internal_node = vv.split_arc(arc, len(cur_extension_suffix))
                self.add_node(new_internal_node, (i, i))
                return new_internal_node


class TestNode(unittest.TestCase):
    def test_equality(self):
        trie = SuffixTree("abcabxabcd")
        random_node = Node(trie)
        self.assertNotEqual(trie.root, random_node)
        self.assertEqual(trie.root, trie.root)
        self.assertTrue(trie.root == trie.root)
        self.assertFalse(trie.root == random_node)


class TestTree(unittest.TestCase):
    def setUp(self):
        # when build_tree is implemented we can remove this.
        self.text = "axabx"
        self.tree = SuffixTree(self.text)
        self.tree.build_tree()

    def test_add_node(self):
        tree = SuffixTree("abcd")
        tree.add_node(tree.root, (0, 1))
        with self.assertRaises(Exception):
            tree.add_node(tree.root, (0, 1))
        self.assertEqual(1, len(tree.root.children))
        tree.add_node(tree.root, (0, 2))
        self.assertEqual(2, len(tree.root.children))

    def check_arc(self, arc, child):
        for gc in child.children.keys():
            if arc[1] < gc[0]:
                continue

            raise Exception("path overlap!")

    def test_integrity(self):
        # build a suffix tree and verify that
        # 1.  from each leaf you can traverse parent pointers and reach the root.
        for l in self.tree.leaves:
            ll = l
            while ll != self.tree.root:
                ll = ll.parent

        # 2.  for each internal node, arc[0] of each outgoing arc is greater than
        #     arc[1] of the incoming arc.

        for item in self.tree.root.children.items():
            self.check_arc(item[0], item[1])

        # 3.  the number of leaves == the number of suffixes.
        self.assertEqual(len(self.text + '$'), len(self.tree.leaves))

    def test_traversal(self):
        self.assertIsNone(self.tree.traverse(self.tree.root, 'cccc'))

        internal_node = self.tree.traverse(self.tree.root, 'a')
        self.assertNotEqual(self.tree.root, internal_node)
        self.assertEqual(internal_node, self.tree.traverse(self.tree.root, 'axa'))

        self.assertEqual(self.tree.root, self.tree.traverse(self.tree.root, 'b'))

        self.assertEqual(self.tree.leaves[0], self.tree.traverse(self.tree.root, 'axabx$'))

        self.assertEqual(internal_node, self.tree.traverse(internal_node, 'b'))

        self.assertEqual(self.tree.leaves[5], self.tree.traverse(self.tree.root, '$'))

        self.assertEqual(self.tree.root, self.tree.traverse(self.tree.root, ''))

    def test_show(self):
        self.tree.show()

    def test_build_tree_no_internal_nodes(self):
        t = SuffixTree("abcd")
        t.build_tree()
        t.show()

    def test_build_tree_one_internal_node(self):
        t = SuffixTree("axa")
        t.build_tree()
        t.show()

    def test_build_tree_axabx(self):
        t = SuffixTree("axabx")
        t.build_tree()
        t.show()

    def test_build_tree_abcabxabcd(self):
        t = SuffixTree("abcabxabcd")
        t.build_tree()
        t.show()

    def test_build_tree_mississippi(self):
        t = SuffixTree("mississippi")
        t.build_tree()
        t.show()

    def test_build_tree_commodore(self):
        t = SuffixTree("commodore")
        t.build_tree()
        t.show()

    def test_build_tree_aaabbb(self):
        t = SuffixTree("aaabbb")
        t.build_tree()
        t.show()

    def test_empty_tree(self):
        # make sure that we can build, traverse, show, and search a suffix tree constructed with the empty string.

        empty = SuffixTree("")
        empty.show()
        pass
