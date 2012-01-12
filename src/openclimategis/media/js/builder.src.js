/*global Ext, google*/
var App, blah;
///////////////////////////////////////////////////////////////////// Overrides
Ext.define('App.ui.ComboBox', {
    override: 'Ext.form.field.ComboBox',
    initialize: function() {
        this.callOverridden(arguments);
        },
    labelWidth: 120
    });
Ext.define('App.ui.Toolbar', {
    override: 'Ext.toolbar.Toolbar',
    initialize: function() {
        this.callOverridden(arguments);
        },
    height: 28
    });
/////////////////////////////////////////////////////////////////////// Classes
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
            new google.maps.Map(this.body.dom, {
                center: new google.maps.LatLng(42.30220, -83.68952),
                zoom: 8,
                mapTypeId: google.maps.MapTypeId.ROADMAP
                });
            // Listen for the 'tilesloaded' event as proxy indicator for 'mapready'
/*
            google.maps.event.addListener(this.gmap, 'tilesloaded', function() {
                self.fireEvent('mapready');
                });
*/
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
    alias: 'widget.treepanel'
    });
Ext.application({
    name: 'App',
    launch: function() {
    /////////////////////////////////////////////////// Application Entry Point
        App.data = {
            stats: Ext.create('Ext.data.TreeStore', {
                sorters: [
                    {property: 'leaf', direction: 'ASC'},
                    {property: 'text', direction: 'ASC'}
                    ],
                root: {
                    expanded: true,
                    children: [
                        {text: 'Basic Statistics', expanded: true,
                            children: [
                                {text: 'Minimum', checked: false, leaf: true},
                                {text: 'Maximum', checked: false, leaf: true},
                                {text: 'Mean', checked: false, leaf: true}
                                ]
                            },
                        {text: 'Thresholds', expanded: true,
                            children: [
                                {text: 'Less than', checked: false, leaf: true},
                                {text: 'Greater than', checked: false, leaf: true},
                                {text: 'Between', checked: false, leaf: true}
                                ]
                            }
                        ] // eo children
                    } // eo root
                })
            };
        //////////////////////////////////////////////////////////// Components
        App.viewport = Ext.create('Ext.container.Viewport', { // Viewport
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
                        {
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
                                    store: App.data.stats,
                                    rootVisible: false,
                                    tbar: [
                                        {
                                            xtype: 'combo',
                                            fieldLabel: 'Grouping Interval',
                                            labelWidth: 100,
                                            queryMode: 'local',
                                            value: 'yearmonth',
                                            valueField: 'value',
                                            store: Ext.create('Ext.data.ArrayStore', {
                                                fields: ['text', 'value'],
                                                data: [
                                                    ['Year and month', 'yearmonth']
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
                    xtype: 'combo',
                    fieldLabel: 'Archive'
                    },
                {
                    xtype: 'combo',
                    fieldLabel: 'Emissions Scenario'
                    },
                {
                    xtype: 'combo',
                    fieldLabel: 'Climate Model'
                    },
                {
                    xtype: 'combo',
                    fieldLabel: 'Variable'
                    },
                {
                    xtype: 'combo',
                    fieldLabel: 'Run'
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
                    width: 200
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


