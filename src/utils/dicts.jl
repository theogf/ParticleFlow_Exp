using ColorSchemes: seaborn_colorblind
colors = seaborn_colorblind

acs = Dict(
    :gpf => "GPF",
    :gf => "GF",
    :dsvi => "DSVI",
    :fcs => "FCS",
    :spm => "SPM",
    :iblr => "IBLR",
    :ngd => "NGD",
)

dcolors = Dict(
    :gpf => colors[1],
    :gf => colors[2],
    :dsvi => colors[3],
    :fcs => colors[4],
    :iblr => colors[5],
    :spm => colors[6],
    )