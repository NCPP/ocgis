Ext.application({
    name: 'App',
    launch: function() {
    /////////////////////////////////////////////////// Application Entry Point
        /////////////////////////////////////////////////////////////// Classes
        Ext.define('App.ui.MarkupComponent', {
            extend: 'Ext.Component',
            alias: 'widget.markup',
            config: {
                frame: false
                }
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
                    layout: 'table',
                    columns: 2,
                    items: [
                        { // Data selection
                            title: 'Data Selection'
                            },
                        { // Temporal selection
                            title: 'Temporal Selection'
                            },
                        { // Output format
                            title: 'Output format'
                            },
                        { // Spatial selection
                            title: 'Spatial Selection',
                            rowspan: 3
                            },
                        { // Data request URL
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
                    collapsed: true
                    },
                ] // eo items
            }); // eo Ext.create
        ////////////////////////////////////////////////////////////////////////
        } // eo launch()
    }); // eo Ext.application


