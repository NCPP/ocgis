Ext.application({
    name: 'App',
    launch: function() {
    /////////////////////////////////////////////////// Application Entry Point
        /////////////////////////////////////////////////////////////// Classes
        Ext.define('App.ui.MarkupComponent', {
            extend: 'Ext.Component',
            alias: 'widget.markup',
            frame: false,
            border: 0
            }); // No callback (third argument)
        Ext.define('App.ui.NestedPanel', {
            extend: 'Ext.Panel',
            alias: 'widget.nested',
            resizable: true
            }); // No callback (third argument)
        //////////////////////////////////////////////////////////// Components
        Ext.create('Ext.container.Viewport', { // Viewport
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
                    region: 'center',
                    layout: 'border',
                    items: [
                        {
                            region: 'west',
                            width: 300,
                            border: 0,
                            layout: 'border',
                            items: [
                                { // Data selection
                                    xtype: 'nested',
                                    title: 'Data Selection',
                                    region: 'north',
                                    height: 200
                                    },
                                { // Temporal selection
                                    xtype: 'nested',
                                    title: 'Temporal Selection',
                                    region: 'center'
                                    },
                                { // Output format
                                    xtype: 'nested',
                                    title: 'Output Format',
                                    region: 'south',
                                    height: 100
                                    }
                                ]
                            },
                        { // Spatial selection
                            title: 'Spatial Selection',
                            region: 'center'
                            },
                        { // Data request URL
                            xtype: 'panel',
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
        ////////////////////////////////////////////////////////////////////////
        } // eo launch()
    }); // eo Ext.application


