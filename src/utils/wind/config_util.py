# @author: ww

from yaml import safe_load
from jinja2 import Template
from collections import UserDict


class Config(UserDict):
    def __init__(self, template_file=None, meta_config=None, **kwargs):
        """
        A Config class for load config from template and render it with meta config
        :param template_file: template file
        :param meta_config: meta config used to fill template, type is dirt
        :param kwargs: some other key-value pair of config,

        >>> m_config = {'date': '2019-11-06', 'pkg': {'python': {'version': '3.6.8'}}}
        >>> print(Config('test_resource/test_config.yml', m_config))
        {'date': datetime.date(2019, 11, 6), 'pkg': {'python': {'version': '3.6.8', 'date': '2019-11-06'}, 'django': {'version': '2.2.6'}}}
        """
        super().__init__(**kwargs)
        self.template_file = template_file
        self.meta_config = meta_config

        self.template_str = ''
        self.config_str = ''
        self.fill(template_file, meta_config)

    def _read_template(self, template_file):
        with open(template_file, 'r') as f:
            self.template_str = f.read()
        return self.template_str

    def _fill_template(self, meta_config):
        self.config_str = Template(self.template_str).render(meta_config)
        return self.config_str

    def _read_config(self, config_str):
        self.update(safe_load(config_str))

    def fill(self, template_file, meta_config=None):
        """
        fill config template which from template file by using meta config
        :param template_file:
        :param meta_config:
        :return:
        """
        if meta_config is None:
            self.meta_config = {}
        else:
            self.meta_config = meta_config
            self.update(meta_config)
        if template_file is None:
            self.template_file = None
        else:
            self.template_file = template_file
            self._read_template(self.template_file)
            self._fill_template(self.meta_config)
            self._read_config(self.config_str)


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    # from file_util import read_yaml
    # m_config = read_yaml('test_resource/test_config.yml')
    # config = Config('test_resource/test_config.yml', m_config)
    # # config.load('test_resource/test_config.yml', m_config)
    # print(config)
