from pathlib import Path

from util import get_args


class Global:
    arg = get_args()
    # external_tools = str(Path.home()) + "/" + arg.global_dir
    # external_tools = str(Path.home()) + "/sampled_data"
    # STANFORD_PARSER = external_tools + '/parse/stanford-parser-full-2018-02-27/stanford-parser.jar'
    # STANFORD_MODELS = external_tools + "/parse/stanford-parser-full-2018-02-27/stanford-parser-3.9.1-models.jar"
