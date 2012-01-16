/*global Ext, google*/
var App, blah;
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
    triggerAction: 'all'
    });
Ext.define('App.ui.Toolbar', {
    override: 'Ext.toolbar.Toolbar',
    initialize: function() {
        this.callOverridden(arguments);
        },
    height: 28
    });
//////////////////////////////////////////////////////////////////////// Models
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
        ]
    });
Ext.define('App.api.Scenario', {
    extend: 'Ext.data.Model',
    fields: ['urlslug', 'description', 'code', 'name', 'id']
    });
Ext.define('App.api.Variable', {
    extend: 'Ext.data.Model',
    fields: ['ndim', 'urlslug', 'code', 'description', 'units', 'id', 'name']
    });
/////////////////////////////////////////////////////////////////////// Classes
Ext.define('App.ui.ApiComboBox', {
    extend: 'Ext.form.field.ComboBox',
    alias: 'widget.apicombo',
    queryMode: 'remote',
    valueField: 'id',
    displayField: 'urlslug',
    onLoad: function(store) { // Arguments: [store, records, success]
        // Set the field to the value of the first record, based on store's valueField
        this.setValue(store.data.items[0].data[this.valueField]);
        }
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
        afterrender: function() {
            var self = this,
                drawingManager = new google.maps.drawing.DrawingManager();
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
            }
        }
    }); // No callback (third argument)
Ext.define('App.ui.DateRange', {
    extend: 'Ext.form.FieldContainer',
    alias: 'widget.daterange',
    msgTarget: 'side',
    layout: 'hbox',
    defaults: {
        width: 90,
        hideLabel: true,
        vtype: 'daterange'
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
        Ext.apply(Ext.form.VTypes, {
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
    rootVisible: true
    });
////////////////////////////////////////////////////////////////////////////////
Ext.application({
    name: 'App',
    launch: function() {
    /////////////////////////////////////////////////// Application Entry Point
        App.data = {
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
                                    itemId: 'output',
                                    title: 'Output Format',
                                    region: 'south',
                                    height: 70
                                    }
                                ]
                            },
                        { // Spatial selection
                            xtype: 'mappanel',
                            itemId: 'map-panel',
                            title: 'Spatial',
                            region: 'center',
                            tbar: [
                                {
                                    xtype: 'combo',
                                    fieldLabel: 'Area-of-Interest (AOI)',
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
                                    width: 180
                                    },
                                {
                                    xtype: 'tbtext',
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
                    store: App.data.archives
                    },
                {
                    xtype: 'apicombo',
                    fieldLabel: 'Emissions Scenario',
                    displayField: 'code',
                    store: App.data.scenarios
                    },
                {
                    xtype: 'combo',
                    fieldLabel: 'Climate Model'
                    },
                {
                    xtype: 'apicombo',
                    fieldLabel: 'Variable',
                    displayField: 'name',
                    store: App.data.variables
                    },
                {
                    xtype: 'numberfield',
                    fieldLabel: 'Run',
                    value: 1,
                    minValue: 1,
                    maxValue: 99
                    },
                {
                    xtype: 'daterange',
                    fieldLabel: 'Date Range',
                    labelWidth: 80,
                    width: 290
                    }
                ]);
            }()); // Execute immediately
        // Add items to the Output Format panel ////////////////////////////////
        (function() {
            var p = Ext.getCmp('form-panel').getComponent('sidebar').getComponent('output');
            p.add([
                {
                    xtype: 'combo',
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
                    emptyText: 'http://openclimategis.org/api/',
                    width: 500,
                    height: 80
                    }
                ]);
            }()); // Execute immediately
        ////////////////////////////////////////////////////////////////////////
        } // eo launch()
    }); // eo Ext.application


