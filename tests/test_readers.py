import unittest
from gbtr.readers.dict_readers import DictReader


class Test_TestDictReader(unittest.TestCase):

    def test_proper_data_reading(self):

        data = [
            {
                'text': 'This is first document!',
                'label': 'Label 1'
            },
            {
                'text': 'This is second document!',
                'label': 'Label 2'
            }
        ]
        reader = DictReader()
        documents = reader.read_data(data)

        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0].text, data[0]['text'])
        self.assertIsInstance(documents[0].text, str)
        self.assertEqual(documents[0].label, data[0]['label'])
        self.assertIsInstance(documents[0].label, str)
        self.assertEqual(documents[1].text, data[1]['text'])
        self.assertIsInstance(documents[1].text, str)
        self.assertEqual(documents[1].label, data[1]['label'])
        self.assertIsInstance(documents[1].label, str)

    def test_other_types_data_reading(self):

        data = [
            {
                'text': 5,
                'label': True
            },
            {
                'text': DictReader,
                'label': False
            }
        ]
        reader = DictReader()
        documents = reader.read_data(data)

        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0].text, str(data[0]['text']))
        self.assertIsInstance(documents[0].text, str)
        self.assertEqual(documents[0].label, str(data[0]['label']))
        self.assertIsInstance(documents[0].label, str)
        self.assertEqual(documents[1].text, str(data[1]['text']))
        self.assertIsInstance(documents[1].text, str)
        self.assertEqual(documents[1].label, str(data[1]['label']))
        self.assertIsInstance(documents[1].label, str)

    def test_missing_data_reading(self):

        data = [
            {
                'label': 'Label 1'
            },
            {
                'text': 'This is second document!',
            }
        ]
        reader = DictReader()
        self.assertRaises(Exception, reader.read_data, data)

    def test_incorrect_data_reading(self):

        reader = DictReader()

        self.assertRaises(Exception, reader.read_data, 123)
        self.assertRaises(Exception, reader.read_data, 'abc')
        self.assertRaises(Exception, reader.read_data, [123])
        self.assertRaises(Exception, reader.read_data, ['abc'])


if __name__ == '__main__':
    unittest.main()
