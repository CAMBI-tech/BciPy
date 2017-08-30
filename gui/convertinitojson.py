import codecs
import collections
import ConfigParser
import json
import os
import sys

objects_list = collections.OrderedDict()
try:
    parser = ConfigParser.SafeConfigParser()
    parserParamNames = ConfigParser.SafeConfigParser()
    parserHelpTips = ConfigParser.SafeConfigParser()
    parserValues = ConfigParser.SafeConfigParser()
    with codecs.open(("static/parameters/listofparams.txt"), 'r', encoding='utf-8') as f:
        with codecs.open(("static/parameters/HumanReadableParamNames.txt"), 'r', encoding='utf-8') as f2:
            with codecs.open(("static/helper_text/parameterhelptips.txt"), 'r', encoding='utf-8') as f3:
                with codecs.open(("static/parameters/recommendedparamvalues.txt"), 'r', encoding='utf-8') as f4:
                    parser.readfp(f)
                    parserParamNames.readfp(f2)
                    parserHelpTips.readfp(f3)
                    parserValues.readfp(f4)
                    for each_section in parser.sections():
                        for (each_key, each_val) in parser.items(each_section):
                            d = collections.OrderedDict()
                            d['value'] = each_val
                            d['section'] = each_section
                            for param_sections in parserParamNames.sections():
                                for(param_key, param_val) in parserParamNames.items(param_sections):
                                    if(each_key == param_key):
                                        d['readableName'] = param_val
                            for help_sections in parserHelpTips.sections():
                                for(help_key, help_val) in parserHelpTips.items(help_sections):
                                    if(each_key == help_key):
                                        d['helpTip'] = help_val
                            for value_sections in parserValues.sections():
                                for(value_key, value_val) in parserValues.items(value_sections):
                                    if(each_key == value_key):
                                        d['recommended_values'] = value_val.split(',')
                            if(not 'recommended_values' in d):
                                d['recommended_values'] = ''
                            objects_list[str(each_key)] = d
    f.close()
    j = json.dumps(objects_list, indent=2)
    if(os.path.isfile('parameters.json')):
        output = open('parameters.json', 'w')
    else:
        output = open('parameters.json', 'a')
    output.write(j)
finally:
    pass
