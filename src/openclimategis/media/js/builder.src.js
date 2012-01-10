Ext.application({
    name: 'App',
    launch: function() {
    /////////////////////////////////////////////////// Application Entry Point
        var viewport;
        ///////////////////////////////////////////////////////////// Overrides
        Ext.define('Override.form.field.ComboBox', {
            override: 'Ext.form.field.ComboBox',
            initialize : function() {
                this.callOverridden(arguments);
                },
            labelWidth: 130
            });
        /////////////////////////////////////////////////////////////// Classes
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
        Ext.define('App.ui.MapPanel', { // Only for the map
            extend: 'Ext.Panel',
            alias: 'widget.map',
            resizable: true
            }); // No callback (third argument)
        Ext.define('App.ui.DateRange', {
            extend: 'Ext.form.FieldContainer',
            alias: 'widget.daterange',
            msgTarget: 'side',
            layout: 'hbox',
            labelWidth: 90,
            width: 290,
            defaults: {
                width: 90,
                hideLabel: true
                },
            items: [
                {
                    xtype: 'datefield',
                    name: 'startDate',
                    itemId: 'date-start',
                    margin: '0 5 0 0'
                    },
                {
                    xtype: 'datefield',
                    name: 'endDate',
                    itemId: 'date-end'
                    }
                ]
            }); // No callback (third argument)
        //////////////////////////////////////////////////////////// Components
        viewport = Ext.create('Ext.container.Viewport', { // Viewport
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
                            itemId: 'form-sidebar',
                            region: 'west',
                            width: 300,
                            border: 0,
                            layout: 'border',
                            items: [
                                { // Data selection
                                    xtype: 'nested',
                                    itemId: 'form-data-sel',
                                    title: 'Data Selection',
                                    region: 'north',
                                    height: 200
                                    },
                                { // Temporal selection
                                    xtype: 'nested',
                                    itemId: 'form-time-sel',
                                    title: 'Temporal',
                                    region: 'center'
                                    },
                                { // Output format
                                    xtype: 'nested',
                                    itemId: 'form-output',
                                    title: 'Output Format',
                                    region: 'south',
                                    height: 100
                                    }
                                ]
                            },
                        { // Spatial selection
                            xtype: 'map',
                            title: 'Spatial',
                            region: 'center',
                            tbar: {
                                defaults: {
                                    style: {fontSize: '11px'}
                                    },
                                items: [
                                    {
                                        xtype: 'combo',
                                        fieldLabel: 'Area-of-Interest (AOI)',
                                        emptyText: '(None Selected)'
                                        },
                                    ' ', // Spacer
                                    {
                                        xtype: 'button',
                                        text: 'Manage AOIs',
                                        iconCls: 'icon-app-edit'
                                        },
                                    '-', // Vertical separator
                                    ' ', // Spacer
                                    {
                                        xtype: 'checkbox',
                                        boxLabel: 'Clip Output to AOI'
                                        },
                                    ' ', // Spacer
                                    '-', // Vertical separator
                                    ' ', // Spacer
                                    {
                                        xtype: 'checkbox',
                                        boxLabel: 'Aggregate Geometries'
                                        }
                                    ] // eo items
                                } // eo tbar
                            },
                        { // Data request URL
                            xtype: 'nested',
                            region: 'south',
                            height: 150,
                            title: 'Data Request URL',
                            colspan: 2
                            }
                        ]
                    },
                { // Help
                    xtype: 'panel',
                    title: 'Help',
                    region: 'east',
                    width: 150,
                    collapsed: true,
                    collapsible: true
                    },
                ] // eo items
            }); // eo Ext.create
        // Add items to the Data Selection panel ///////////////////////////////
        (function() {
            var p = Ext.getCmp('form-panel').getComponent('form-sidebar').getComponent('form-data-sel');
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
                    fieldLabel: 'Date Range'
                    },
                ]);
            }()); // Execute immediately
        // Add items to the Temporal Selection panel ///////////////////////////
        (function() {
            var p = Ext.getCmp('form-panel').getComponent('form-sidebar').getComponent('form-time-sel');
            p.add([
                {
                    xtype: 'combo',
                    fieldLabel: 'Grouping Interval'
                    }
                ]);
            }()); // Execute immediately
        // Add items to the Output Format panel ////////////////////////////////
        (function() {
            var p = Ext.getCmp('form-panel').getComponent('form-sidebar').getComponent('form-output');
            p.add([
                {
                    xtype: 'combo',
                    labelAlign: 'top',
                    fieldLabel: 'Output Format'
                    }
                ]);
            }()) // Execute immediately
        ////////////////////////////////////////////////////////////////////////
        } // eo launch()
    }); // eo Ext.application


