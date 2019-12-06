#!/usr/bin/env python

import logging
import sys
import unittest
import fileinput


def get_logger():
    logger = logging.getLogger('ukkonen')

    formatter = logging.Formatter('%(asctime)s - %(filename)s:%(funcName)s:%(lineno)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    level = logging.INFO
    logger.setLevel(level)
    ch.setLevel(level)

    return logger


logger = get_logger()

MARKER = 0xfeedface


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
        assert(0 <= arc[0] < len(self.tree.text))
        assert(arc[0] <= arc[1] < len(self.tree.text) or arc[1] == MARKER)

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

        # this forces arc label to be at least 2 chars.
        assert(0 < n)
        assert(arc[1] == MARKER or n < arc[1] - arc[0] + 1)

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

        arc_len = self.tree.arc_length(arc)

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

    def get_arc_label(self, arc):
        if arc[1] == MARKER:
            return self.text[arc[0]:]
        return self.text[arc[0]:arc[1] + 1]

    def show(self):
        print self.text
        self.root.show()

    @property
    def text(self):
        return self.__text

    def is_substring(self, s):
        current_node = self.root
        i = 0
        t = current_node.get_key(s[i])
        while t:
            min_l = min(t[1] - t[0] + 1, len(s[i:]))
            label = self.get_arc_label(t)

            if label[:min_l] != s[i:i + min_l]:
                return False

            i += min_l
            if i == len(s):
                return True

            current_node = current_node.children[t]
            t = current_node.get_key(s[i])

        return False

    def is_suffix(self, s):
        return self.is_substring(s + '$')

    def suffix_helper(self, partial_string, node):
        if node.is_leaf():
            yield partial_string
            return

        # this sorts the arcs by first character on the label
        keys = sorted(node.children.keys(), key=lambda x: self.text[x[0]])

        for k in keys:
            for x in self.suffix_helper(partial_string + self.get_arc_label(k), node.children[k]):
                yield x

    def suffixes(self):
        for s in self.suffix_helper('', self.root):
            yield s

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

    def fix_leaves(self):
        """
        after the tree is built, change the arcs into each leaf so that arc[1] is not MARKER but the index of
        the last char in the text.
        :return:
        """
        for l in self.leaves:
            p = l.parent
            arc = l.get_arc_from_parent()
            assert arc is not None
            new_arc = (arc[0], len(self.text) - 1)
            del p.children[arc]
            p.children[new_arc] = l

    def build_tree(self):
        # add the first character to construct T1.

        # phase and extension numbers are 1-based.

        logger.debug("phase 1 - constructing T1")
        self.add_node(self.root, (0, MARKER))

        m = len(self.text)
        i = 1
        while i < m:
            current_char = self.text[i]

            nleaves = len(self.leaves)

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
                    skip_remaining_extensions, new_internal_node = self.apply_extension_rules(cur_extension_suffix, current_char, i, v)
                else:
                    sv = v.suffix_link
                    if sv == self.root:
                        skip_remaining_extensions, new_internal_node = self.apply_extension_rules(cur_extension_suffix, current_char, i, sv)
                    else:
                        partial_path = self.partial_path(current_node, v)
                        assert(prev_extension_suffix.startswith(partial_path))
                        gamma = prev_extension_suffix[len(partial_path):]
                        skip_remaining_extensions, new_internal_node = self.apply_extension_rules(gamma, current_char, i, sv)

                if new_internal_node:
                    if last_internal_node:
                        last_internal_node.suffix_link = new_internal_node
                    new_internal_node.suffix_link = self.root
                    last_internal_node = new_internal_node
                if skip_remaining_extensions:
                    break
                j += 1
            i += 1

        self.fix_leaves()
        logger.debug("suffix tree complete; %s leaves" % len(self.leaves))

    def apply_extension_rules(self, cur_extension_suffix, current_char, i, v):
        """

        :param cur_extension_suffix:
        :param current_char:
        :param i:
        :param v: node from which w
        :return: tuple (boolean, Node).  the boolean indicates whether rule 3 was applied.  if it was, than we can
        skip the remaining extensions in the current phase.  the Node in the pair is the internal node, if any, that
        was created by applying the extension rules.  otherwise None is returned as the node value.
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
                return True, None

            # have to add new leaf
            self.add_node(vv, (i, MARKER))
            return False, None

        # find the end of cur_extension_suffix.
        arc = vv.get_key(cur_extension_suffix[0])
        label = self.get_arc_label(arc)
        # since vv is root, we know that cur_extension_suffix does not end at another node.  therefore
        # the label is at least 1 char longer than cur_extension_suffix.
        if label.startswith(cur_extension_suffix + current_char):
            # cur_extension_suffix + current_char is already here:  rule 3
            return True, None

        # have to split
        new_internal_node = vv.split_arc(arc, len(cur_extension_suffix))
        self.add_node(new_internal_node, (i, MARKER))
        return False, new_internal_node

    def arc_length(self, arc):
        if arc[1] == MARKER:
            return len(self.text) - arc[0]
        return arc[1] - arc[0] + 1


if __name__ == '__main__':
    fi = fileinput.FileInput()
    line = fi.readline().strip()
    t = SuffixTree(line)
    t.build_tree()
    t.show()


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

        self.assertTrue(t.is_substring('abcabxabcd'))
        self.assertTrue(t.is_substring('abcabxabc'))
        self.assertTrue(t.is_substring('abcabxab'))
        self.assertTrue(t.is_substring('abcabxa'))
        self.assertTrue(t.is_substring('abcabx'))
        self.assertTrue(t.is_substring('abcab'))
        self.assertTrue(t.is_substring('abca'))
        self.assertTrue(t.is_substring('abc'))
        self.assertTrue(t.is_substring('ab'))
        self.assertTrue(t.is_substring('a'))

        self.assertTrue(t.is_substring('bcabxabcd'))
        self.assertTrue(t.is_substring('cabxabcd'))
        self.assertTrue(t.is_substring('abxabcd'))
        self.assertTrue(t.is_substring('bxabcd'))
        self.assertTrue(t.is_substring('xabcd'))
        self.assertTrue(t.is_substring('abcd'))
        self.assertTrue(t.is_substring('bcd'))
        self.assertTrue(t.is_substring('cd'))
        self.assertTrue(t.is_substring('d'))

        self.assertTrue(t.is_substring('b'))
        self.assertTrue(t.is_substring('c'))
        self.assertTrue(t.is_substring('x'))

        self.assertFalse(t.is_substring('xba'))
        self.assertFalse(t.is_substring('poop'))
        self.assertFalse(t.is_substring('aa'))
        self.assertFalse(t.is_substring('bcx'))
        self.assertFalse(t.is_substring('abcabxabce'))
        self.assertFalse(t.is_substring('abdc'))

        ######

        self.assertTrue(t.is_suffix('abcabxabcd'))
        self.assertFalse(t.is_suffix('abcabxabc'))
        self.assertFalse(t.is_suffix('abcabxab'))
        self.assertFalse(t.is_suffix('abcabxa'))
        self.assertFalse(t.is_suffix('abcabx'))
        self.assertFalse(t.is_suffix('abcab'))
        self.assertFalse(t.is_suffix('abca'))
        self.assertFalse(t.is_suffix('abc'))
        self.assertFalse(t.is_suffix('ab'))
        self.assertFalse(t.is_suffix('a'))

        self.assertTrue(t.is_suffix('bcabxabcd'))
        self.assertTrue(t.is_suffix('cabxabcd'))
        self.assertTrue(t.is_suffix('abxabcd'))
        self.assertTrue(t.is_suffix('bxabcd'))
        self.assertTrue(t.is_suffix('xabcd'))
        self.assertTrue(t.is_suffix('abcd'))
        self.assertTrue(t.is_suffix('bcd'))
        self.assertTrue(t.is_suffix('cd'))
        self.assertTrue(t.is_suffix('d'))

        self.assertFalse(t.is_suffix('b'))
        self.assertFalse(t.is_suffix('c'))
        self.assertFalse(t.is_suffix('x'))

    def test_build_tree_mississippi(self):
        t = SuffixTree("mississippi")
        t.build_tree()
        t.show()

        for s in t.suffixes():
            print s

    def test_build_tree_commodore(self):
        t = SuffixTree("commodore")
        t.build_tree()
        t.show()

    def test_build_tree_aaabbb(self):
        t = SuffixTree("aaabbb")
        t.build_tree()
        t.show()

    def test_build_tree_anbn(self):
        n = 23
        text = 'a' * n + 'b' * n
        t = SuffixTree(text)
        t.build_tree()
        t.show()

        for s in t.suffixes():
            print s

    def test_empty_tree(self):
        # make sure that we can build, traverse, show, and search a suffix tree constructed with the empty string.

        empty = SuffixTree("")
        empty.show()
        all_suffixes = [x for x in empty.suffixes()]
        self.assertEqual(1, len(all_suffixes))
