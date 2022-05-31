# Experiments and Fields

BciPy experiments and fields allow users to collect metadata that may be integrated alongside core BciPy data. This is particularly useful when the experiment has completed and a researcher wants to curate their files for sharing with the community.

## Experiments

An experiment defines the name, summary and fields to collect. At this level, the requirement for collection during every task and whether or not to anonymize later will be set. The anonymization does not encrypt the data at rest, but instead defines how to handle the field later when uploading/sharing. The registered experiments are defined in `.bcipy/experiment/experiments.json` in the following format:

```js
{ 
    name: { 
        fields : {
            name: "",
            required: bool,
            anonymize: bool
        },
        summary: "" 
        }
}
```

## Fields

A field is a unit of data collection for an experiment. It has a name, help text and type. The type will determine how it is collected and validated. The registered fields are defined in `.bcipy/field/fields.json` in the following format:


```js
{
    name: {
        help_text: "",
        type: "FieldType"
        }
}
```

where FieldType may be str, bool, int, or float.