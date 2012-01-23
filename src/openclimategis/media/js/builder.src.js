/*global Ext, google*/
var App, blah = [], bloo = [];
/*
Ext.require([
    'Ext.Component',
    'Ext.container.Viewport',
    'Ext.data.Model',
    'Ext.data.reader.Json',
    'Ext.data.Store',
    'Ext.data.TreeStore',
    'Ext.form.FieldContainer',
    'Ext.form.field.*',
    'Ext.Panel',
    'Ext.toolbar.Toolbar',
    'Ext.tree.*'
    ]);
*/
///////////////////////////////////////////////////////////////////// Overrides
Ext.define('App.ui.BaseField', {
    override: 'Ext.form.field.Base',
    initialize: function() {
        this.callOverridden(arguments);
        },
    labelWidth: 120,
    triggerAction: 'all',
    isQueryParam: true, // NOTE: All form fields are assumed to represent query parameters
    listeners: {
        change: function(field, newValue) { // Third argument is "oldValue"
            if (this.isQueryParam) {
                this.findParentByType('form').fireEvent('change', {field: field, newValue: newValue});
                }
            } // eo change
        } // eo listeners
    });
Ext.define('App.ui.BaseContainer', {
    override: 'Ext.container.Container',
    initialize: function() {
        this.callOverridden(arguments);
        },
    getValues: function() {
        var results = {},
            fields = this.query("{getValue}"); // Assumption is only fields have getValue member
        Ext.each(fields, function(i) {
            results[i.name] = i.getValue();
            });
        return results;
        }
    });
Ext.define('App.ui.Toolbar', {
    override: 'Ext.toolbar.Toolbar',
    initialize: function() {
        this.callOverridden(arguments);
        },
    height: 28
    });
//////////////////////////////////////////////////////////////////////// Models
Ext.define('App.api.AreaOfInterest', {
    extend: 'Ext.data.Model',
    fields: ['uid_field', 'code', {name: 'id', type: 'int'}, 'desc']
    });
Ext.define('App.api.Archive', {
    extend: 'Ext.data.Model',
    fields: ['code', {name: 'id', type: 'int'}, 'name', 'url', 'urlslug']
    });
Ext.define('App.api.Function', {
    extend: 'Ext.data.Model',
    fields: [
        'text',
        {name: 'leaf', type: 'boolean'},
        {name: 'children', type: 'auto'},
        'value',
        'desc'
        ],
    hasFormatString: function() {
        if (this.get('desc').indexOf('{0}') > 0) {
            return true;
            }
        return false;
        },
    singleValued: function() {
        if (this.get('desc').indexOf('{1}') > 0) {
            return false;
            }
        return true;
        },
    getComponents: function() {
        var raw = this.get('desc');
        if (this.singleValued()) {
            if (raw.indexOf('{0}') > 0) { // Clip to the {0} placeholder
                this.set('first', raw.substr(0, raw.indexOf('{0}')));
                }
            return [
                {
                    xtype: 'textfield',
                    fieldLabel: this.get('first'),
                    labelAlign: 'top',
                    labelSeparator: '',
                    isQueryParam: false,
                    vtype: 'numeric'
                    }
                ]; // eo return
            } // eo if
        else {
            this.set('first', raw.substr(0, raw.indexOf('{0}')));
            this.set('second', raw.substring(raw.indexOf('{0}') + 3, raw.indexOf('{1}')));
            return [
                {
                    xtype: 'inlinetextfield',
                    fieldLabel: this.get('first'),
                    name: 'first',
                    isQueryParam: false,
                    vtype: 'numeric'
                    },
                {
                    xtype: 'inlinetextfield',
                    fieldLabel: this.get('second'),
                    name: 'second',
                    isQueryParam: false,
                    vtype: 'numeric'
                    }
                ]; // eo return
            } // eo else
        } // eo getComponents
    }, function() {
        Ext.apply(Ext.form.field.VTypes, {
            numeric: function(val) {
                var re = /^\d*\.?\d*$/; // Matches numbers only
                return re.test(val);
                },
            numericText: 'Only numeric values are allowed'
            });
        });
Ext.define('App.api.Model', {
    extend: 'Ext.data.Model',
    fields: ['urlslug', 'code', 'organization', 'id', 'name', 'comments']
    });
Ext.define('App.api.Scenario', {
    extend: 'Ext.data.Model',
    fields: ['urlslug', 'code', 'description', 'id', 'name']
    });
Ext.define('App.api.Variable', {
    extend: 'Ext.data.Model',
    fields: ['ndim', 'units', 'urlslug', 'code', 'description', 'id', 'name']
    });
/////////////////////////////////////////////////////////////////////// Classes
Ext.define('App.ui.ApiComboBox', {
    extend: 'Ext.form.field.ComboBox',
    alias: 'widget.apicombo',
    queryMode: 'remote',
    valueField: 'urlslug',
    displayField: 'urlslug',
    onLoad: function(store) { // Arguments: [store, records, success]
        // Set the field to the value of the first record, based on store's valueField
        this.setValue(store.data.items[0].data[this.valueField]);
        }
    });
Ext.define('App.ui.OkButton', {
    extend: 'Ext.button.Button',
    alias: 'widget.ok',
    width: 69,
    text: 'OK',
    style: {margin: '5px 3px 0'},
    handler: function() {
        // The callback expects the button "type" (e.g. 'ok') and the form values
        if (!this.callback(this.text.toLowerCase(), this.ownerCt.getValues())) {
            this.findParentByType('window').close();
            }
        }
    });
Ext.define('App.ui.CancelButton', {
    extend: 'Ext.button.Button',
    alias: 'widget.cancel',
    width: 69,
    text: 'Cancel',
    style: {margin: '5px 3px 0'},
    handler: function() {
        // The callback expects the button "type" (e.g. 'ok') and its value
        if (!this.callback(this.text.toLowerCase(), this.ownerCt.getValues())) {
            this.findParentByType('window').close();
            }
        }
    });
Ext.define('App.ui.InlineTextField', {
    extend: 'Ext.form.field.Text',
    alias: 'widget.inlinetextfield',
    labelSeparator: '',
    labelAlign: 'top',
    fieldStyle: {width: 80}
    });
Ext.define('App.ui.MarkupComponent', { // No ExtJS fluff
    extend: 'Ext.Component',
    alias: 'widget.markup',
    frame: false,
    border: 0
    }); // No callback (third argument)
Ext.define('App.ui.Container', { // Children should only be panels
    extend: 'Ext.Panel',
    alias: 'widget.container',
    resizable: true
    }); // No callback (third argument)
Ext.define('App.ui.NestedPanel', { // Padded bodies
    extend: 'Ext.Panel',
    alias: 'widget.nested',
    resizable: true,
    bodyPadding: 7
    }); // No callback (third argument)
/*
Ext.define('App.ui.MapPanel', {
    extend: 'Ext.Panel',
    alias: 'widget.mappanel',
    initComponent : function(){
        var config = {
            layout: 'fit',
            mapConfig: {
                center: new google.maps.LatLng(42.30220, -83.68952),
                zoom: 8,
                type: google.maps.MapTypeId.ROADMAP
                }
            };
        Ext.applyIf(this, config);
        this.callParent();        
        },
    listeners: {
        render: function() {
            this.body.mask();
            },
        afterrender: function() {
            var self = this,
                modes = google.maps.drawing.OverlayType,
                drawingManager = new google.maps.drawing.DrawingManager({
                    drawingControlOptions: {
                        drawingModes: [modes.RECTANGLE, modes.POLYGON]
                        }
                    });
            this.gmap = new google.maps.Map(this.body.dom, {
                center: new google.maps.LatLng(42.30220, -83.68952),
                zoom: 8,
                mapTypeId: google.maps.MapTypeId.ROADMAP
                });
            drawingManager.setMap(this.gmap);
            // Listen for the 'tilesloaded' event as proxy indicator for 'mapready'
            google.maps.event.addListener(this.gmap, 'tilesloaded', function() {
                self.fireEvent('mapready');
                });
            },
        mapready: function() {
            this.body.unmask();
            }
        }
    }); // No callback (third argument)
*/
Ext.define('App.ui.DateRange', {
    extend: 'Ext.form.FieldContainer',
    alias: 'widget.daterange',
    msgTarget: 'side',
    layout: 'hbox',
    outFormatEach: 'Y-m-d',
    outFormat: '{0}+{1}', // e.g. Output format is 'Y-m-d+Y-m-d'
    defaults: {
        width: 90,
        hideLabel: true,
        vtype: 'daterange',
        isQueryParam: false,
        listeners: {
            change: function() {
                if (this.prev()) {
                    if (this.prev().getValue() === null) {
                        return;
                        }
                    }
                else if (this.next()) {
                    if (this.next().getValue() === null) {
                        return;
                        }
                    }
                this.findParentByType('form').fireEvent('change', {
                    field: {
                        name: this.ownerCt.name || 'temporal',
                        isQueryParam: this.ownerCt.isQueryParam || true
                        },
                    newValue: this.ownerCt.getValue()
                    });
                }
            }
        },
    getValue: function() {
        var d1, d2, v = this.getValues();
        // The outFormatEach property should follow Ext.Date formatting
        d1 = Ext.util.Format.date(v.startDate, this.outFormatEach);
        d2 = Ext.util.Format.date(v.endDate, this.outFormatEach);
        return Ext.String.format(this.outFormat, d1, d2); // e.g. this.outFormat = '{0} to {1}'
        },
    items: [
        {
            xtype: 'datefield',
            name: 'startDate',
            itemId: 'date-start',
            endDateField: 'date-end',
            emptyText: 'Start',
            margin: '0 5 0 0'
            },
        {
            xtype: 'datefield',
            name: 'endDate',
            itemId: 'date-end',
            startDateField: 'date-start',
            emptyText: 'End'
            }
        ]
    }, function() { // Callback function
        Ext.apply(Ext.form.field.VTypes, {
            daterange: function(val, field) {
                var date = field.parseDate(val), start, end;
                if (!date) {
                    return false;
                    }
                if (field.startDateField) {
                    start = field.ownerCt.getComponent(field.startDateField);
                    if (!start.maxValue || (date.getTime() !== start.maxValue.getTime())) {
                        start.setMaxValue(date);
                        start.validate();
                        }
                    }
                else if (field.endDateField) {
                    end = field.ownerCt.getComponent(field.endDateField);
                    if (!end.minValue || (date.getTime() !== end.minValue.getTime())) {
                        end.setMinValue(date);
                        end.validate();
                        }
                    }
                /**
                 * Always return true since we're only using this vtype to set the
                 * min/max allowed values (these are tested for after the vtype test)
                 */
                return true;
                }
            });
        });
Ext.define('App.ui.TreePanel', {
    extend: 'Ext.tree.Panel',
    alias: 'widget.treepanel',
    rootVisible: true,
    getValue: function() {
        var stats = this.getChecked(),
            value = [];
        Ext.each(stats, function(i, n, all) {
            if (i.get('attrs')) {
                value.push({
                    value: i.get('value'),
                    attrs: i.get('attrs')
                    });
                }
            else {value.push(i.get('value'));}
            });
        return value;
        },
    listeners: {
        beforeitemmousedown: function(view, rec) {
            var cb = function(btn, values) {
                    if (btn === 'cancel') {
                        rec.set('checked', !rec.get('checked'));
                        } 
                    else {
                        Ext.iterate(values, function(k, v, o) {
                            if (!Ext.isNumeric(v)) {
                                Ext.Msg.alert('Invalid Value', 'You must enter a numeric value only.').setIcon(Ext.Msg.ERROR);
                                rec.set('checked', !rec.get('checked'));
                                } // eo if
                            else {
                                // Adds the user-defined values to the record
                                rec.set('attrs', values);
                                }
                            }); // eo Ext.iterate()
                        } // eo else
                    },
                prompt;
            // Is the box already checked?
            if (rec.get('checked')) { // Uncheck the box
                rec.set('checked', !rec.get('checked'));
                }
            else { // Check the box
                rec.set('checked', !rec.get('checked'));
                // Is inline formatting needed?
                if (rec.hasFormatString()) {
                    prompt = Ext.create('Ext.Window', {
                        title: 'Add Statistic: ' + rec.get('text'),
                        buttons: Ext.Msg.OKCANCEL,
                        closable: false,
                        animateTarget: this.header.getEl(),
                        modal: true,
                        bodyPadding: 10,
                        width: 250,
                        items: rec.getComponents(),
                        style: {textAlign: 'center'}
                        });
                    prompt.add([
                        {xtype: 'ok', callback: cb},
                        {xtype: 'cancel', callback: cb}
                        ]);
                    prompt.show();
                    } // eo if
                } // eo else
            } // eo itemclick
        } // eo listeners
    });
////////////////////////////////////////////////////////////////////////////////
Ext.application({
    name: 'App',
    launch: function() {
        Ext.getBody().mask('Loading...');
    /////////////////////////////////////////////////// Application Entry Point
        App.data = {
            aois: Ext.create('Ext.data.Store', {
                model: 'App.api.AreaOfInterest',
                proxy: {
                    type: 'ajax',
                    url: '/api/aois.json',
                    reader: 'json'
                    }
                }),
            archives: Ext.create('Ext.data.Store', {
                model: 'App.api.Archive',
                proxy: {
                    type: 'ajax',
                    url: '/api/archives.json',
                    reader: 'json'
                    }
                }),
            functions: Ext.create('Ext.data.TreeStore', {
                model: 'App.api.Function',
                proxy: {
                    type: 'ajax',
                    url: '/api/functions.json',
                    reader: 'json'
                    },
                sorters: [
                    {property: 'leaf', direction: 'ASC'},
                    {property: 'text', direction: 'ASC'}
                    ],
                root: {
                    text: 'Available Statistics',
                    expanded: true
                    }
                }),
            models: Ext.create('Ext.data.Store', {
                model: 'App.api.Model',
                proxy: {
                    type: 'ajax',
                    url: '/api/models.json',
                    reader: 'json'
                    }
                }),
            outputs: Ext.create('Ext.data.ArrayStore', {
                fields: ['value', 'text'],
                data: [
                    ['geojson', 'GeoJSON Text File'],
                    ['csv', 'Comma Separated Value'],
                    ['kcsv', 'Linked Comma Separated Value (zipped)'],
                    ['shz', 'ESRI Shapefile (zipped)'],
                    ['lshz', 'CSV-Linked ESRI Shapefile (zipped)'],
                    ['kml', 'Keyhole Markup Language'],
                    ['kmz', 'Keyhole Markup Language (zipped)'],
                    ['sqlite', 'SQLite3 Database (zipped)'],
                    ['nc', 'NetCDF']
                    ] 
                }),
            scenarios: Ext.create('Ext.data.Store', {
                model: 'App.api.Scenario',
                proxy: {
                    type: 'ajax',
                    url: '/api/scenarios.json',
                    reader: 'json'
                    }
                }),
            variables: Ext.create('Ext.data.Store', {
                model: 'App.api.Variable',
                proxy: {
                    type: 'ajax',
                    url: '/api/variables.json',
                    reader: 'json'
                    }
                })
            };
        //////////////////////////////////////////////////////////// Components
        App.viewport = Ext.create('Ext.container.Viewport', {
            id: 'viewport',
            layout: 'border',
            listeners: {
                afterrender: function() {
                    Ext.getBody().unmask();
                    }
                },
            items: [
                { // Banner
                    xtype: 'markup',
                    region: 'north',
                    height: 50,
                    style: {
                        background: '#000'
                        },
                    html: '<div id="branding"><a href="/">OpenClimateGIS&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</a></div>'
                    },
                { // Form
                    xtype: 'form',
                    id: 'form-panel',
                    region: 'center',
                    layout: 'border',
                    listeners: {
                        change: function(args) { // Changes the API Request URL display
                            if (args.field.isQueryParam) {
                                Ext.getCmp('form-api-url').updateUrl(args.field.name, args.newValue);
                                } // eo if
                            }
                        },
                    items: [
                        { // Sidebar
                            xtype: 'container',
                            itemId: 'sidebar',
                            region: 'west',
                            width: 310,
                            border: 0,
                            layout: 'border',
                            items: [
                                { // Data selection
                                    xtype: 'nested',
                                    itemId: 'data-selection',
                                    title: 'Data Selection',
                                    region: 'north',
                                    height: 200
                                    },
                                { // Temporal selection
                                    xtype: 'treepanel',
                                    itemId: 'stats-tree',
                                    title: 'Temporal',
                                    region: 'center',
                                    store: App.data.functions,
                                    tbar: [
                                        {
                                            xtype: 'combo',
                                            fieldLabel: 'Grouping Interval',
                                            name: 'grouping',
                                            labelWidth: 100,
                                            queryMode: 'local',
                                            value: 'year',
                                            valueField: 'value',
                                            width: 200,
                                            store: Ext.create('Ext.data.ArrayStore', {
                                                fields: ['text', 'value'],
                                                data: [
                                                    ['Year', 'year'],
                                                    ['Month', 'month'],
                                                    ['Day', 'day']
                                                    ] // data
                                                }) // eo store
                                            } // eo combo
                                        ] // eo tbar
                                    },
                                { // Output format
                                    xtype: 'nested',
                                    itemId: 'output-format',
                                    title: 'Output Format',
                                    region: 'south',
                                    height: 70
                                    }
                                ]
                            },
                        { // Spatial selection
                            xtype: 'panel',
                            itemId: 'map-panel',
                            title: 'Spatial',
                            region: 'center',
                            tbar: [
                                {
                                    xtype: 'combo',
                                    fieldLabel: 'Area-of-Interest (AOI)',
                                    name: 'aoi',
                                    isQueryParam: false, // TODO Is this right?
                                    store: App.data.aois,
                                    displayField: 'code',
                                    valueField: 'id',
                                    emptyText: '(None Selected)',
                                    style: {textAlign: 'right'}
                                    },
                                ' ',
                                {
                                    xtype: 'button',
                                    text: 'Manage AOIs',
                                    iconCls: 'icon-app-edit'
                                    },
                                {
                                    xtype: 'button',
                                    text: 'Clip Output to AOI',
                                    iconCls: 'icon-scissors',
                                    enableToggle: true
                                    },
                                {
                                    xtype: 'button',
                                    text: 'Aggregate Geometries',
                                    iconCls: 'icon-shape-group',
                                    enableToggle: true
                                    },
                                '->',
                                {
                                    xtype: 'button',
                                    text: 'Save Sketch As AOI',
                                    iconCls: 'icon-disk'
                                    }
                                ] // eo items
                            },
                        { // Data request URL
                            xtype: 'nested',
                            itemId: 'request-url',
                            region: 'south',
                            height: 150,
                            title: 'Data Request URL',
                            bbar: [
                                {
                                    xtype: 'button',
                                    iconCls: 'icon-page-do',
                                    text: 'Generate Data File'
                                    },
                                {
                                    xtype: 'progressbar',
                                    id: 'query-progress',
                                    width: 180
                                    },
                                {
                                    xtype: 'tbtext',
                                    id: 'query-status',
                                    text: 'No activity',
                                    style: {fontStyle: 'italic'}
                                    }
                                ] // eo items
                            } // eo nested
                        ] // eo items
                    },
                { // Help
                    xtype: 'panel',
                    title: 'Help',
                    region: 'east',
                    width: 150,
                    collapsed: true,
                    collapsible: true
                    }
                ] // eo items
            }); // eo Ext.create
        // Add items to the Data Selection panel ///////////////////////////////
        (function() {
            var p = Ext.getCmp('form-panel').getComponent('sidebar').getComponent('data-selection');
            p.add([
                {
                    xtype: 'apicombo',
                    fieldLabel: 'Archive',
                    name: 'archive',
                    store: App.data.archives
                    },
                {
                    xtype: 'apicombo',
                    fieldLabel: 'Climate Model',
                    name: 'model',
                    displayField: 'code',
                    store: App.data.models
                    },
                {
                    xtype: 'apicombo',
                    fieldLabel: 'Emissions Scenario',
                    name: 'scenario',
                    displayField: 'code',
                    store: App.data.scenarios
                    },
                {
                    xtype: 'apicombo',
                    fieldLabel: 'Variable',
                    name: 'variable',
                    displayField: 'name',
                    store: App.data.variables
                    },
                {
                    xtype: 'numberfield',
                    fieldLabel: 'Run',
                    name: 'run',
                    value: 1,
                    minValue: 1,
                    maxValue: 99
                    },
                {
                    xtype: 'daterange',
                    fieldLabel: 'Date Range',
                    name: 'temporal',
                    labelWidth: 80,
                    width: 290
                    }
                ]);
            }()); // Execute immediately
        // Add items to the Output Format panel ////////////////////////////////
        (function() {
            var p = Ext.getCmp('form-panel').getComponent('sidebar').getComponent('output-format');
            p.add([
                {
                    xtype: 'combo',
                    name: 'format',
                    width: 250,
                    queryMode: 'local',
                    valueField: 'value',
                    value: 'geojson',
                    store: App.data.outputs
                    }
                ]);
            }()); // Execute immediately
        // Add items to the Data Request URL panel /////////////////////////////
        (function() {
            var p = Ext.getCmp('form-panel').getComponent('request-url');
            p.add([
                {
                    xtype: 'textarea',
                    name: 'query',
                    id: 'form-api-url',
                    isQueryParam: false,
                    width: 500,
                    height: 80,
                    cls: 'updateable', // Indicates this field should flash when updated
                    fieldBodyCls: 'shell', // Indicates this field's body text is code to be copy/pasted
                    value: 'http://openclimategis.org/api/',
                    template: 'http://openclimategis.org/api/archive/{0}/model/{1}/scenario/{2}/run/{3}/temporal/{4}/spatial/{5}/aggregate/{6}/variable/{7}.{8}',
                    // This is a holder for the parameter values
                    values: {
                        archive: '{0}',
                        model: '{1}',
                        scenario: '{2}',
                        run: '{3}',
                        temporal: '{4}', 
                        spatial: '{5}',
                        aggregate: '{6}',
                        variable: '{7}',
                        format: '{8}'
                        },
                    listeners: {
                        afterrender: function() {
                            Ext.apply(this.values, Ext.getCmp('form-panel').getValues());
                            },
                        change: function() { // Draw some attention to this box
                            Ext.getCmp('form-api-url').container.highlight();
                            this.animate({
                                duration: 400,
                                easing: 'backIn',
                                from: {opacity: 0.4},
                                to: {opacity: 1}
                                });
                            /* Other choices for animation
                            this.getEl().slideIn('t', {
                                easing: 'easeIn',
                                duration: 300
                                })
                            */
                            }
                        },
                    updateUrl: function(name, value) {
                        var s, v = this.values;
                        this.values[name] = value; // Set the updated parameter's value
                        s = Ext.String.format(this.template, v.archive, v.model, v.scenario, v.run, v.temporal, v.spatial, v.aggregate, v.variable, v.format);
/*
                        // If-else clause here basically just tries to hide format characters (e.g. '{1}') as much as possible
                        if (this.getValue() === s.substring(0, s.indexOf('{')) && this.getValue().length <= s.substring(0, s.indexOf('{'))) {
                            this.setValue(s.substring(0, s.lastIndexOf('{')));
                            }
                        else {
                            this.setValue(s.substring(0, s.indexOf('{')));
                            }
*/
                        this.setValue(s.substring(0, s.indexOf('{')));
                        return this.getValue(); // Return the updated URL
                        } // eo updateUrl()
                    } // eo textarea
                ]);
            }()); // Execute immediately
        ////////////////////////////////////////////////////////////////////////
        } // eo launch()
    }); // eo Ext.application


