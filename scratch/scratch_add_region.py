import sys
import os
import re

def generate_region(class_name, properties):
    # properties is list of tuples: (Type, Name)
    
    # lower case first letter of name for param name
    params = []
    for t, n in properties:
        param_name = n[0].lower() + n[1:]
        params.append((t, n, param_name))
        
    param_sig = ", ".join([f"{t} {pn}" for t, n, pn in params])
    param_sig_with_input = f"ISeries<double> input, " + param_sig if params else "ISeries<double> input"
    param_pass = ", ".join([pn for t, n, pn in params])
    param_pass_with_input = "Input, " + param_pass if params else "Input"
    param_pass_with_input_lower = "input, " + param_pass if params else "input"
    
    cache_checks = " && ".join([f"cache_{class_name}[idx].{n} == {pn}" for t, n, pn in params])
    cache_checks = " && " + cache_checks if cache_checks else ""
    
    init_props = ", ".join([f"{n} = {pn}" for t, n, pn in params])
    
    code = f"""
#region NinjaScript generated code. Neither change nor remove.

namespace NinjaTrader.NinjaScript.Indicators
{{
\tpublic partial class Indicator : NinjaTrader.Gui.NinjaScript.IndicatorRenderBase
\t{{
\t\tprivate {class_name}[] cache_{class_name};
\t\tpublic {class_name} {class_name}({param_sig})
\t\t{{
\t\t\treturn {class_name}({param_pass_with_input});
\t\t}}

\t\tpublic {class_name} {class_name}({param_sig_with_input})
\t\t{{
\t\t\tif (cache_{class_name} != null)
\t\t\t\tfor (int idx = 0; idx < cache_{class_name}.Length; idx++)
\t\t\t\t\tif (cache_{class_name}[idx] != null{cache_checks} && cache_{class_name}[idx].EqualsInput(input))
\t\t\t\t\t\treturn cache_{class_name}[idx];
\t\t\treturn CacheIndicator<{class_name}>(new {class_name}(){{ {init_props} }}, input, ref cache_{class_name});
\t\t}}
\t}}
}}

namespace NinjaTrader.NinjaScript.MarketAnalyzerColumns
{{
\tpublic partial class MarketAnalyzerColumn : MarketAnalyzerColumnBase
\t{{
\t\tpublic Indicators.{class_name} {class_name}({param_sig})
\t\t{{
\t\t\treturn indicator.{class_name}({param_pass_with_input});
\t\t}}

\t\tpublic Indicators.{class_name} {class_name}({param_sig_with_input})
\t\t{{
\t\t\treturn indicator.{class_name}({param_pass_with_input_lower});
\t\t}}
\t}}
}}

namespace NinjaTrader.NinjaScript.Strategies
{{
\tpublic partial class Strategy : NinjaTrader.Gui.NinjaScript.StrategyRenderBase
\t{{
\t\tpublic Indicators.{class_name} {class_name}({param_sig})
\t\t{{
\t\t\treturn indicator.{class_name}({param_pass_with_input});
\t\t}}

\t\tpublic Indicators.{class_name} {class_name}({param_sig_with_input})
\t\t{{
\t\t\treturn indicator.{class_name}({param_pass_with_input_lower});
\t\t}}
\t}}
}}

#endregion
"""
    return code

files = [
    ("1a-StatCloseRegressionBands_v1.0-RC.cs", "_1a_StatCloseRegressionBands_v10", [
        ("BarsPeriodType", "CloseTimeFrameType"),
        ("int", "CloseTimeFrameValue"),
        ("int", "Period"),
        ("bool", "ShowSigma1"),
        ("bool", "ShowSigma2"),
        ("bool", "ShowSigma3"),
        ("bool", "ShowSigma4"),
    ]),
    ("1b-StatHlRegressionBands_v1.0-RC.cs", "_1b_StatHlRegressionBands_v10", [
        ("BarsPeriodType", "HlTimeFrameType"),
        ("int", "HlTimeFrameValue"),
        ("int", "HlPeriod"),
        ("bool", "ShowHighFarSide"),
        ("bool", "ShowHighNearSide"),
        ("bool", "ShowLowFarSide"),
        ("bool", "ShowLowNearSide"),
        ("bool", "ShowSigma1"),
        ("bool", "ShowSigma2"),
        ("bool", "ShowSigma3"),
        ("bool", "ShowSigma4"),
    ]),
    ("2-CubicRegressionEndpoint_v1.0-RC.cs", "_2_CubicRegressionEndpoint_v10", [
        ("BarsPeriodType", "CubicTimeFrameType"),
        ("int", "CubicTimeFrameValue"),
        ("int", "Period"),
    ])
]

base_dir = r"C:\Users\reyse\OneDrive\Documents\NinjaTrader 8\bin\Custom\Indicators"

for file_name, class_name, props in files:
    file_path = os.path.join(base_dir, file_name)
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if "#region NinjaScript generated code" in content:
        print(f"Skipping {file_name} (already has region)")
        continue
        
    # Inject before the last closing brace
    # First, strip trailing whitespace
    content = content.rstrip()
    if content.endswith("}"):
        content = content[:-1]
        if content.rstrip().endswith("}"):
            content = content.rstrip()[:-1]
            
            region_code = generate_region(class_name, props)
            new_content = content + "\n" + region_code + "\n}\n"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Injected region into {file_name}")
        else:
            print(f"Format unexpected for {file_name}")
    else:
        print(f"Format unexpected for {file_name}")
